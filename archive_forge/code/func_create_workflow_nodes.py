from __future__ import absolute_import, division, print_function
from ..module_utils.controller_api import ControllerAPIModule
import json
def create_workflow_nodes(module, response, workflow_nodes, workflow_id):
    for workflow_node in workflow_nodes:
        workflow_node_fields = {}
        search_fields = {}
        association_fields = {}
        if workflow_node['unified_job_template']['name']:
            if workflow_node['unified_job_template']['type'] is None:
                module.fail_json(msg='Could not find unified job template type in workflow_nodes {0}'.format(workflow_node))
            search_fields['type'] = workflow_node['unified_job_template']['type']
            if workflow_node['unified_job_template']['type'] == 'inventory_source':
                if 'inventory' in workflow_node['unified_job_template']:
                    if 'organization' in workflow_node['unified_job_template']['inventory']:
                        organization_id = module.resolve_name_to_id('organizations', workflow_node['unified_job_template']['inventory']['organization']['name'])
                        search_fields['organization'] = organization_id
                    else:
                        pass
            elif 'organization' in workflow_node['unified_job_template']:
                organization_id = module.resolve_name_to_id('organizations', workflow_node['unified_job_template']['organization']['name'])
                search_fields['organization'] = organization_id
            else:
                pass
            unified_job_template = module.get_one('unified_job_templates', name_or_id=workflow_node['unified_job_template']['name'], **{'data': search_fields})
            if unified_job_template:
                workflow_node_fields['unified_job_template'] = unified_job_template['id']
            elif workflow_node['unified_job_template']['type'] != 'workflow_approval':
                module.fail_json(msg='Unable to Find unified_job_template: {0}'.format(search_fields))
        for field_name in ('identifier', 'extra_data', 'scm_branch', 'job_type', 'job_tags', 'skip_tags', 'limit', 'diff_mode', 'verbosity', 'forks', 'job_slice_count', 'timeout', 'all_parents_must_converge', 'state'):
            field_val = workflow_node.get(field_name)
            if field_val:
                workflow_node_fields[field_name] = field_val
            if workflow_node['identifier']:
                search_fields = {'identifier': workflow_node['identifier']}
            if 'execution_environment' in workflow_node:
                workflow_node_fields['execution_environment'] = module.get_one('execution_environments', name_or_id=workflow_node['execution_environment']['name'])['id']
        if 'inventory' in workflow_node:
            if 'name' in workflow_node['inventory']:
                inv_lookup_data = {}
                if 'organization' in workflow_node['inventory']:
                    inv_lookup_data['organization'] = module.resolve_name_to_id('organizations', workflow_node['inventory']['organization']['name'])
                workflow_node_fields['inventory'] = module.get_one('inventories', name_or_id=workflow_node['inventory']['name'], data=inv_lookup_data)['id']
            else:
                workflow_node_fields['inventory'] = module.get_one('inventories', name_or_id=workflow_node['inventory'])['id']
        search_fields['workflow_job_template'] = workflow_node_fields['workflow_job_template'] = workflow_id
        existing_item = module.get_one('workflow_job_template_nodes', **{'data': search_fields})
        state = True
        if 'state' in workflow_node:
            if workflow_node['state'] == 'absent':
                state = False
        if state:
            response.append(module.create_or_update_if_needed(existing_item, workflow_node_fields, endpoint='workflow_job_template_nodes', item_type='workflow_job_template_node', auto_exit=False))
        else:
            response.append(module.delete_if_needed(existing_item, auto_exit=False))
        if workflow_node['unified_job_template']['type'] == 'workflow_approval':
            for field_name in ('name', 'description', 'timeout'):
                field_val = workflow_node['unified_job_template'].get(field_name)
                if field_val:
                    workflow_node_fields[field_name] = field_val
            workflow_job_template_node = module.get_one('workflow_job_template_nodes', **{'data': search_fields})
            workflow_job_template_node_id = workflow_job_template_node['id']
            existing_item = None
            if workflow_job_template_node['related'].get('unified_job_template') is not None:
                existing_item = module.get_endpoint(workflow_job_template_node['related']['unified_job_template'])['json']
            approval_endpoint = 'workflow_job_template_nodes/{0}/create_approval_template/'.format(workflow_job_template_node_id)
            module.create_or_update_if_needed(existing_item, workflow_node_fields, endpoint=approval_endpoint, item_type='workflow_job_template_approval_node', associations=association_fields, auto_exit=False)
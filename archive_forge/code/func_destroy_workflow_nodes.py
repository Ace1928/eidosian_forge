from __future__ import absolute_import, division, print_function
from ..module_utils.controller_api import ControllerAPIModule
import json
def destroy_workflow_nodes(module, response, workflow_id):
    search_fields = {}
    search_fields['workflow_job_template'] = workflow_id
    existing_items = module.get_all_endpoint('workflow_job_template_nodes', **{'data': search_fields})
    for workflow_node in existing_items['json']['results']:
        response.append(module.delete_endpoint(workflow_node['url']))
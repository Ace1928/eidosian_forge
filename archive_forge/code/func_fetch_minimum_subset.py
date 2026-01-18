from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
import re
def fetch_minimum_subset(info_subset):
    if info_subset is None:
        return ({}, True)
    minimum_subset = ['arrays', 'disks', 'folders', 'groups', 'initiator_groups', 'performance_policies', 'pools', 'protection_schedules', 'protection_templates', 'protocol_endpoints', 'snapshot_collections', 'software_versions', 'users', 'volumes', 'volume_collections']
    toreturn = {'default': {}}
    result = {}
    temp_dict = {}
    try:
        for key in minimum_subset:
            cl_obj = info_subset[key]
            if key == 'arrays':
                resp = cl_obj.list(detail=True, fields='extended_model,full_name,all_flash')
            elif key == 'groups':
                if utils.is_array_version_above_or_equal(info_subset['arrays'], '5.1'):
                    resp = cl_obj.list(detail=True, fields='encryption_config,name,fc_enabled,iscsi_enabled,leader_array_name,default_iscsi_target_scope,num_snaps')
                else:
                    resp = cl_obj.list(detail=True, fields='name')
            else:
                resp = cl_obj.list(detail=False)
            temp_dict[key] = resp
        result['volumes'] = len(temp_dict['volumes'])
        result['volume_collections'] = len(temp_dict['volume_collections'])
        result['users'] = len(temp_dict['users'])
        result['software_versions'] = temp_dict['software_versions'][-1].attrs.get('version')
        result['snapshot_collections'] = len(temp_dict['snapshot_collections'])
        result['snapshots'] = temp_dict['groups'][-1].attrs.get('num_snaps')
        result['protocol_endpoints'] = len(temp_dict['protocol_endpoints'])
        result['protection_templates'] = len(temp_dict['protection_templates'])
        result['protection_schedules'] = len(temp_dict['protection_schedules'])
        result['initiator_groups'] = len(temp_dict['initiator_groups'])
        result['folders'] = len(temp_dict['folders'])
        result['disks'] = len(temp_dict['disks'])
        result['folders'] = len(temp_dict['folders'])
        result['arrays'] = generate_dict('arrays', temp_dict['arrays'])['arrays']
        result['groups'] = generate_dict('groups', temp_dict['groups'])['groups']
        toreturn['default'] = result
        return (toreturn, True)
    except Exception as ex:
        result['failed'] = str(ex)
        toreturn['default'] = result
        return (toreturn, False)
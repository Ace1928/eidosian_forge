from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_device_ids_from_group_names(module, rest_obj):
    try:
        grp_name_list = module.params.get('device_group_names')
        resp = rest_obj.get_all_report_details(group_service_path)
        group_id_list = []
        grp_list_resp = resp['report_list']
        if grp_list_resp:
            for name in grp_name_list:
                for group in grp_list_resp:
                    if group['Name'] == name:
                        group_id_list.append(group['Id'])
                        break
        else:
            module.exit_json(msg='Unable to fetch the specified device_group_names.', baseline_compliance_info=[])
        return get_device_ids_from_group_ids(module, group_id_list, rest_obj)
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err
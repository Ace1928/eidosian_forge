from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_device_ids_from_group_ids(module, grou_id_list, rest_obj):
    try:
        device_id_list = []
        for group_id in grou_id_list:
            group_id_path = group_service_path + '({group_id})/Devices'.format(group_id=group_id)
            resp_val = rest_obj.get_all_items_with_pagination(group_id_path)
            grp_list_value = resp_val['value']
            if grp_list_value:
                for device_item in grp_list_value:
                    device_id_list.append(device_item['Id'])
        if len(device_id_list) == 0:
            module.exit_json(msg='Unable to fetch the device ids from specified device_group_names.', baseline_compliance_info=[])
        return device_id_list
    except (URLError, HTTPError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err
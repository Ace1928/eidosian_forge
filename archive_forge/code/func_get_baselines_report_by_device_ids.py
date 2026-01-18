from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_baselines_report_by_device_ids(rest_obj, module):
    try:
        device_ids, identifier = get_identifiers(rest_obj, module)
        if device_ids or identifier == 'device_ids':
            resp = rest_obj.invoke_request('POST', baselines_report_by_device_ids_path, data={'Ids': device_ids})
            return resp.json_data
        else:
            identifier_map = {'device_group_names': 'Device details not available as the group name(s) provided are invalid.', 'device_service_tags': 'Device details not available as the service tag(s) provided are invalid.'}
            message = identifier_map[identifier]
            module.exit_json(msg=message)
    except HTTPError as err:
        err_message = json.load(err)
        err_list = err_message.get('error', {}).get('@Message.ExtendedInfo', [{'Message': EXIT_MESSAGE}])
        if err_list:
            err_reason = err_list[0].get('Message', EXIT_MESSAGE)
            if MSG_ID in err_list[0].get('MessageId'):
                module.exit_json(msg=err_reason)
        raise err
    except (URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err
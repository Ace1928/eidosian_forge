from __future__ import (absolute_import, division, print_function)
import json
import copy
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import MANAGER_JOB_ID_URI, wait_for_redfish_reboot_job, \
def fetch_storage_resource(module, session_obj):
    try:
        system_uri = '{0}{1}'.format(session_obj.root_uri, 'Systems')
        system_resp = session_obj.invoke_request('GET', system_uri)
        system_members = system_resp.json_data.get('Members')
        if system_members:
            system_id_res = system_members[0]['@odata.id']
            SYSTEM_ID = system_id_res.split('/')[-1]
            system_id_res_resp = session_obj.invoke_request('GET', system_id_res)
            system_id_res_data = system_id_res_resp.json_data.get('Storage')
            if system_id_res_data:
                storage_collection_map.update({'storage_base_uri': system_id_res_data['@odata.id']})
            else:
                module.fail_json(msg='Target out-of-band controller does not support storage feature using Redfish API.')
        else:
            module.fail_json(msg='Target out-of-band controller does not support storage feature using Redfish API.')
    except HTTPError as err:
        if err.code in [404, 405]:
            module.fail_json(msg='Target out-of-band controller does not support storage feature using Redfish API.', error_info=json.load(err))
        raise err
    except (URLError, SSLValidationError, ConnectionError, TypeError, ValueError) as err:
        raise err
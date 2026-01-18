from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
def fetch_session_inactivity_settings(rest_obj):
    final_resp = rest_obj.invoke_request('GET', SESSION_INACTIVITY_GET)
    ret_data = final_resp.json_data.get('value')
    return ret_data
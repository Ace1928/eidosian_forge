from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params, \
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_redfish_reboot_job, \
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def require_session(idrac, module):
    session_id, token = ('', None)
    payload = {'UserName': module.params['username'], 'Password': module.params['password']}
    path = SESSION_RESOURCE_COLLECTION['SESSION']
    resp = idrac.invoke_request('POST', path, data=payload, api_timeout=120)
    if resp and resp.success:
        session_id = resp.json_data.get('Id')
        token = resp.headers.get('X-Auth-Token')
    return (session_id, token)
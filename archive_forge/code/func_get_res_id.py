from __future__ import (absolute_import, division, print_function)
import json
import base64
import os
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import reset_idrac
def get_res_id(idrac, cert_type):
    cert_map = {'Server': MANAGER_ID}
    try:
        resp = idrac.invoke_request(cert_map.get(cert_type, MANAGERS_URI), 'GET')
        membs = resp.json_data.get('Members')
        res_uri = membs[0].get('@odata.id')
        res_id = res_uri.split('/')[-1]
    except Exception:
        res_id = cert_map.get(cert_type, MANAGER_ID)
    return res_id
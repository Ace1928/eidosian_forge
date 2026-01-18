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
def get_ssl_payload(module, operation, cert_type):
    payload = {}
    method = 'POST'
    if operation == 'import':
        payload = _build_import_payload(module, cert_type)
    elif operation == 'export':
        payload = {'SSLCertType': cert_type}
    elif operation == 'generate_csr':
        payload = _build_generate_csr_payload(module, cert_type)
    elif operation == 'reset':
        payload = '{}'
    return (payload, method)
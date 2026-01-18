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
def exit_certificates(module, idrac, cert_url, cert_payload, method, cert_type, res_id):
    cmd = module.params.get('command')
    changed = changed_map.get(cmd)
    reset = changed_map.get(cmd) and module.params.get('reset')
    result = {'changed': changed}
    reset_msg = ''
    if changed:
        reset_msg = 'Reset iDRAC to apply the new certificate. Until the iDRAC is reset, the old certificate will remain active.'
    if module.params.get('command') == 'import':
        export_cert = get_export_data(idrac, cert_type, res_id)
        if cert_payload.get('SSLCertificateFile') in export_cert:
            module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode and changed:
        module.exit_json(msg=CHANGES_MSG, changed=changed)
    if module.params.get('command') == 'reset' and cert_type == 'Server':
        resp = idrac.invoke_request(cert_url, method, data=cert_payload, dump=False)
    else:
        resp = idrac.invoke_request(cert_url, method, data=cert_payload)
    cert_data = resp.json_data
    cert_output = format_output(module, cert_data)
    result.update(cert_output)
    if reset:
        reset, track_failed, reset_msg = reset_idrac(idrac, module.params.get('wait'), res_id)
    if cmd == 'import' and cert_type == 'Server' and module.params.get('ssl_key'):
        result['msg'] = '{0} {1}'.format(SUCCESS_MSG_SSL.format(command=cmd), reset_msg)
    else:
        result['msg'] = '{0}{1}'.format(SUCCESS_MSG.format(command=cmd), reset_msg)
    module.exit_json(**result)
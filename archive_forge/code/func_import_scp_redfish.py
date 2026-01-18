from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def import_scp_redfish(module, idrac, http_share):
    import_buffer = module.params.get('import_buffer')
    command = module.params['command']
    scp_targets = ','.join(module.params['scp_components'])
    perform_check_mode(module, idrac, http_share)
    share = {}
    if not import_buffer:
        share, _scp_file_name_format = get_scp_share_details(module)
        share['file_name'] = module.params.get('scp_file')
        buffer_text = get_buffer_text(module, share)
        share_dict = share
        if share['share_type'] == 'LOCAL':
            share_dict = {}
        idrac_import_scp_params = {'import_buffer': buffer_text, 'target': scp_targets, 'share': share_dict, 'job_wait': module.params['job_wait'], 'host_powerstate': module.params['end_host_power_state'], 'shutdown_type': module.params['shutdown_type']}
        scp_response = idrac.import_scp_share(**idrac_import_scp_params)
        scp_response = wait_for_job_tracking_redfish(module, idrac, scp_response)
    else:
        scp_response = idrac.import_scp(import_buffer=import_buffer, target=scp_targets, job_wait=module.params['job_wait'])
    scp_response = response_format_change(scp_response, module.params, share.get('file_name'))
    exit_on_failure(module, scp_response, command)
    return scp_response
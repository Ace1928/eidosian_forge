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
def preview_scp_redfish(module, idrac, http_share, import_job_wait=False):
    import_buffer = module.params.get('import_buffer')
    command = module.params['command']
    scp_targets = 'ALL'
    job_wait_option = module.params['job_wait']
    if command == 'import':
        job_wait_option = import_job_wait
    share = {}
    if not import_buffer:
        if http_share:
            share_url = urlparse(module.params['share_name'])
            share = {'share_ip': share_url.netloc, 'share_name': share_url.path.strip('/'), 'share_type': share_url.scheme.upper(), 'file_name': module.params.get('scp_file'), 'username': module.params.get('share_user'), 'password': module.params.get('share_password')}
            if http_share == 'HTTPS':
                share['ignore_certificate_warning'] = IGNORE_WARNING[module.params['ignore_certificate_warning']]
        else:
            share, _scp_file_name_format = get_scp_share_details(module)
            share['file_name'] = module.params.get('scp_file')
        buffer_text = get_buffer_text(module, share)
        scp_response = idrac.import_preview(import_buffer=buffer_text, target=scp_targets, share=share, job_wait=False)
        scp_response = wait_for_job_tracking_redfish(module, idrac, scp_response)
    else:
        scp_response = idrac.import_preview(import_buffer=import_buffer, target=scp_targets, job_wait=job_wait_option)
    scp_response = response_format_change(scp_response, module.params, share.get('file_name'))
    exit_on_failure(module, scp_response, command)
    return scp_response
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
def export_scp_redfish(module, idrac):
    command = module.params['command']
    share, scp_file_name_format = get_scp_share_details(module)
    scp_components = ','.join(module.params['scp_components'])
    include_in_export = IN_EXPORTS[module.params['include_in_export']]
    if share['share_type'] == 'LOCAL':
        scp_response = idrac.export_scp(export_format=module.params['export_format'], export_use=module.params['export_use'], target=scp_components, include_in_export=include_in_export, job_wait=False, share=share)
        scp_response = wait_for_response(scp_response, module, share, idrac)
    else:
        scp_response = idrac.export_scp(export_format=module.params['export_format'], export_use=module.params['export_use'], target=scp_components, include_in_export=include_in_export, job_wait=False, share=share)
        scp_response = wait_for_job_tracking_redfish(module, idrac, scp_response)
    scp_response = response_format_change(scp_response, module.params, scp_file_name_format)
    exit_on_failure(module, scp_response, command)
    return scp_response
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
def perform_check_mode(module, idrac, http_share=True):
    if module.check_mode:
        module.params['job_wait'] = True
        scp_resp = preview_scp_redfish(module, idrac, http_share, import_job_wait=True)
        if 'SYS081' in scp_resp['MessageId'] or 'SYS082' in scp_resp['MessageId']:
            module.exit_json(msg=CHANGES_FOUND, changed=True)
        elif 'SYS069' in scp_resp['MessageId']:
            module.exit_json(msg=NO_CHANGES_FOUND)
        else:
            module.fail_json(msg=scp_resp)
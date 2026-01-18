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
def exit_on_failure(module, scp_response, command):
    if isinstance(scp_response, dict) and (scp_response.get('TaskStatus') == 'Critical' or scp_response.get('JobState') in ('Failed', 'CompletedWithErrors')):
        module.fail_json(msg=FAIL_MSG.format(command), scp_status=scp_response)
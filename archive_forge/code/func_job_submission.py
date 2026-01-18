from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def job_submission(self, job_name, job_desc, targets, params, job_type, schedule='startnow', state='Enabled'):
    job_payload = {'JobName': job_name, 'JobDescription': job_desc, 'Schedule': schedule, 'State': state, 'Targets': targets, 'Params': params, 'JobType': job_type}
    response = self.invoke_request('POST', JOB_SERVICE_URI, data=job_payload)
    return response
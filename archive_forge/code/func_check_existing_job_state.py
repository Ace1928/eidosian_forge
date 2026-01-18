from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.common.parameters import env_fallback
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def check_existing_job_state(self, job_type_name):
    query_param = {'$filter': 'LastRunStatus/Id eq 2030 or LastRunStatus/Id eq 2040 or LastRunStatus/Id eq 2050'}
    job_resp = self.invoke_request('GET', JOB_SERVICE_URI, query_param=query_param)
    job_lst = job_resp.json_data['value'] if job_resp.json_data.get('value') is not None else []
    for job in job_lst:
        if job['JobType']['Name'] == job_type_name:
            job_allowed = False
            available_jobs = job
            break
    else:
        job_allowed = True
        available_jobs = job_lst
    return (job_allowed, available_jobs)
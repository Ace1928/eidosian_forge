from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_scheduled_job_resp(idrac_obj, job_type):
    job_resp = {}
    job_list = idrac_obj.invoke_request(MANAGER_JOB_URI, 'GET').json_data.get('Members', [])
    for each_job in job_list:
        if each_job.get('JobType') == job_type and each_job.get('JobState') in ['Scheduled', 'Running', 'Starting']:
            job_resp = each_job
            break
    return job_resp
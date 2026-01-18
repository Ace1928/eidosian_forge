from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def idrac_system_reset(idrac, res_id, payload=None, job_wait=True, wait_time_sec=300, interval=30):
    track_failed, reset, job_resp = (True, False, {})
    reset_msg = RESET_UNTRACK
    try:
        idrac.invoke_request(SYSTEM_RESET_URI.format(res_id=res_id), 'POST', data=payload)
        time.sleep(10)
        if wait_time_sec:
            resp = idrac.invoke_request(MANAGER_JOB_URI, 'GET')
            job = list(filter(lambda d: d['JobState'] in ['RebootPending'], resp.json_data['Members']))
            if job:
                job_resp, msg = wait_for_idrac_job_completion(idrac, MANAGER_JOB_ID_URI.format(job[0]['Id']), job_wait=job_wait, wait_timeout=wait_time_sec)
                if 'job is not complete' in msg:
                    reset, reset_msg = (False, msg)
                if not msg:
                    reset = True
    except Exception:
        reset = False
        reset_msg = RESET_FAIL
    return (reset, track_failed, reset_msg, job_resp)
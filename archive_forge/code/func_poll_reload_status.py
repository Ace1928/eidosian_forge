from __future__ import (absolute_import, division, print_function)
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.memset import memset_api_call
def poll_reload_status(api_key=None, job_id=None, payload=None):
    """
    We poll the `job.status` endpoint every 5 seconds up to a
    maximum of 6 times. This is a relatively arbitrary choice of
    timeout, however requests rarely take longer than 15 seconds
    to complete.
    """
    memset_api, stderr, msg = (None, None, None)
    payload['id'] = job_id
    api_method = 'job.status'
    _has_failed, _msg, response = memset_api_call(api_key=api_key, api_method=api_method, payload=payload)
    while not response.json()['finished']:
        counter = 0
        while counter < 6:
            sleep(5)
            _has_failed, msg, response = memset_api_call(api_key=api_key, api_method=api_method, payload=payload)
            counter += 1
    if response.json()['error']:
        stderr = 'Reload submitted successfully, but the Memset API returned a job error when attempting to poll the reload status.'
    else:
        memset_api = response.json()
        msg = None
    return (memset_api, msg, stderr)
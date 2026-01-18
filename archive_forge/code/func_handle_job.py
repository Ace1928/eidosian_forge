from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import \
def handle_job(module, rest_obj, job_id):
    if module.params.get('job_wait'):
        job_failed, msg, job_dict, wait_time = job_tracking(rest_obj, JOB_URI.format(job_id=job_id), max_job_wait_sec=module.params.get('job_wait_timeout'))
        try:
            job_resp = rest_obj.invoke_request('GET', LAST_EXEC.format(job_id=job_id))
            msg = job_resp.json_data.get('Value')
            msg = msg.replace('\n', ' ')
        except Exception:
            msg = job_dict.get('JobDescription', msg)
        module.exit_json(failed=job_failed, msg=msg, job_id=job_id, changed=True)
    else:
        module.exit_json(changed=True, msg=APPLY_TRIGGERED, job_id=job_id)
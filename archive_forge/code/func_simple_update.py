from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params, \
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_redfish_reboot_job, \
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def simple_update(redfish_obj, preview_uri, update_uri):
    job_ids = []
    for uri in preview_uri:
        resp = redfish_obj.invoke_request('POST', update_uri, data={'ImageURI': uri})
        time.sleep(30)
        task_uri = resp.headers.get('Location')
        task_id = task_uri.split('/')[-1]
        job_ids.append(task_id)
    return job_ids
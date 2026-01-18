from __future__ import (absolute_import, division, print_function)
import json
import time
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import remove_key
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def exit_catalog(module, rest_obj, catalog_resp, operation, msg):
    if module.params.get('job_wait'):
        job_failed, job_message = rest_obj.job_tracking(catalog_resp.get('TaskId'), job_wait_sec=module.params['job_wait_timeout'], sleep_time=JOB_POLL_INTERVAL)
        catalog = get_updated_catalog_info(module, rest_obj, catalog_resp)
        if job_failed is True:
            module.fail_json(msg=job_message, catalog_status=catalog)
        catalog_resp = catalog
        msg = CATALOG_UPDATED.format(operation=operation)
    time.sleep(SETTLING_TIME)
    catalog = get_updated_catalog_info(module, rest_obj, catalog_resp)
    module.exit_json(msg=msg, catalog_status=remove_key(catalog), changed=True)
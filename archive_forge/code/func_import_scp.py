from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.parameters import env_fallback
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import config_ipv6
def import_scp(self, import_buffer=None, target=None, job_wait=False):
    """
        This method imports system configuration details to the system.
        :param import_buffer: import buffer payload content xml or json format
        :param target: IDRAC or NIC or ALL or BIOS or RAID.
        :param job_wait: True or False decide whether to wait till the job completion.
        :return: json response
        """
    payload = {'ImportBuffer': import_buffer, 'ShareParameters': {'Target': target}}
    response = self.invoke_request(IMPORT_URI, 'POST', data=payload)
    if response.status_code == 202 and job_wait:
        task_uri = response.headers['Location']
        response = self.wait_for_job_complete(task_uri, job_wait=job_wait)
    return response
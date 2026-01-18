from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.urls import urlparse
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def check_task_on_device(self, path):
    uri = 'https://{0}:{1}{2}'.format(self.client.provider['server'], self.client.provider['server_port'], path)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201, 202] or ('code' in response and response['code'] in [200, 201, 202]):
        return response
    raise F5ModuleError(resp.content)
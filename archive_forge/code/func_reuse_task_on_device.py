from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def reuse_task_on_device(self, task):
    if task == 'discovery':
        uri = 'https://{0}:{1}/mgmt/cm/global/tasks/device-discovery'.format(self.client.provider['server'], self.client.provider['server_port'])
    else:
        uri = 'https://{0}:{1}/mgmt/cm/global/tasks/device-import'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = "?$filter=deviceReference/link%20eq%20'{0}'".format(self.device_id)
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 409]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    if 'items' in response:
        if response['items']:
            self.task_id = response['id']
            return True
    return False
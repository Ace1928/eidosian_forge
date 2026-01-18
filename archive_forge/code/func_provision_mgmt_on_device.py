from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def provision_mgmt_on_device(self, params):
    uri = 'https://{0}:{1}/mgmt/tm/sys/db/provision.extramb/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.patch(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 404]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def apply_on_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/asm/tasks/apply-policy/'.format(self.client.provider['server'], self.client.provider['server_port'])
    params = dict(policyReference={'link': self.have.self_link})
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return response['id']
    raise F5ModuleError(resp.content)
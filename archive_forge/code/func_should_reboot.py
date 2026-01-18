from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def should_reboot(self):
    for x in range(0, 24):
        try:
            uri = 'https://{0}:{1}/mgmt/tm/sys/db/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], 'provision.action')
            resp = self.client.api.get(uri)
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in response and response['code'] in [400, 404]:
                if 'message' in response:
                    raise F5ModuleError(response['message'])
                else:
                    raise F5ModuleError(resp.content)
            if response['value'] == 'reboot':
                return True
            elif response['value'] == 'none':
                time.sleep(5)
        except Exception:
            time.sleep(5)
    return False
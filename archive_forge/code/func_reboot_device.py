from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def reboot_device(self):
    nops = 0
    last_reboot = self._get_last_reboot()
    try:
        params = dict(command='run', utilCmdArgs='-c "/sbin/reboot"')
        uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 403]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
        if 'commandResult' in response:
            return str(response['commandResult'])
    except Exception:
        pass
    time.sleep(20)
    while nops < 3:
        try:
            self.client.reconnect()
            next_reboot = self._get_last_reboot()
            if next_reboot is None:
                nops = 0
            if next_reboot == last_reboot:
                nops = 0
            else:
                nops += 1
        except Exception:
            pass
        time.sleep(10)
    return None
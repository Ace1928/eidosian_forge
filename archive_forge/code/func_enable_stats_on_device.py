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
def enable_stats_on_device(self):
    params = dict(enabled=self.want.stats_enabled, pushIntervalSecs=self.want.interval, zone=self.want.zone, modules=self.want.stat_modules, targetDeviceReference=dict(link='https://localhost/mgmt/cm/system/machineid-resolver/{0}'.format(self.device_id)))
    uri = 'https://{0}:{1}/mgmt/cm/shared/stats-mgmt/agent-install-and-config-task'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 409]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    task = 'https://{0}:{1}/mgmt/cm/shared/stats-mgmt/agent-install-and-config-task/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], response['id'])
    query = '?$select=status,currentStep,errorMessage'
    self._wait_for_task(task + query)
    return True
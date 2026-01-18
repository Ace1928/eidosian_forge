from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def install_on_device(self):
    try:
        params = dict(command='run', utilCmdArgs='-c "{0}"'.format(self.want.install_command))
        uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.post(uri, json=params)
        try:
            output = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in output and output['code'] in [400, 403]:
            if 'message' in output:
                raise F5ModuleError(output['message'])
            else:
                raise F5ModuleError(resp.content)
    except Exception as ex:
        if 'Connection aborted' in str(ex):
            pass
        elif 'TimeoutException' in str(ex):
            pass
        elif 'remoteSender' in str(ex):
            pass
        else:
            raise F5ModuleError(str(ex))
    self.wait_for_rest_api_restart()
    self.wait_for_configuration_reload()
    return True
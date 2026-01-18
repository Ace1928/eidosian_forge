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
def check_file_exists_on_device(self):
    params = dict(command='run', utilCmdArgs='/var/config/rest/downloads/{0}'.format(self.want.package_file))
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-ls'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        if 'commandResult' in response:
            if 'No such file or directory' in response['commandResult']:
                return False
            elif self.want.package_file in response['commandResult']:
                return True
    raise F5ModuleError(resp.content)
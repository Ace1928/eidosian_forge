from __future__ import absolute_import, division, print_function
import os
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def remove_temp_file_from_device(self):
    name = os.path.split(self.want.source)[1]
    tpath_name = '/var/config/rest/downloads/{0}'.format(name)
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-rm/'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs=tpath_name)
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)
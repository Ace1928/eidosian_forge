from __future__ import absolute_import, division, print_function
from datetime import datetime
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def move_on_device(self, remote_path):
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-mv'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs='{0} /tmp/{1}'.format(remote_path, os.path.basename(remote_path)))
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)
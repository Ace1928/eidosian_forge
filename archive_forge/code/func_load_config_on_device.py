from __future__ import absolute_import, division, print_function
import os
import tempfile
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import dumps
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def load_config_on_device(self, name):
    filepath = '/var/config/rest/downloads/{0}'.format(name)
    command = 'imish -r {0} -f {1}'.format(self.want.route_domain, filepath)
    params = {'command': 'run', 'utilCmdArgs': '-c "{0}"'.format(command)}
    uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        if 'commandResult' in response:
            if 'Dynamic routing is not enabled' in response['commandResult']:
                raise F5ModuleError(response['commandResult'])
        return True
    raise F5ModuleError(resp.content)
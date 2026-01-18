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
def execute_show_commands(self, commands):
    body = []
    uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    for command in to_list(commands):
        command = 'imish -r {0} -e \\"{1}\\"'.format(self.want.route_domain, command)
        params = {'command': 'run', 'utilCmdArgs': '-c "{0}"'.format(command)}
        resp = self.client.api.post(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        if 'commandResult' in response:
            if 'Dynamic routing is not enabled' in response['commandResult']:
                raise F5ModuleError(response['commandResult'])
            body.append(response['commandResult'])
    return body
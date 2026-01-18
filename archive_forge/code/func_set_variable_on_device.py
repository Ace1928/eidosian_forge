from __future__ import absolute_import, division, print_function
import os
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def set_variable_on_device(self, commands):
    command = ' '.join(['user-defined {0} \\"{1}\\"'.format(k, v) for k, v in iteritems(commands)])
    command = 'tmsh modify gtm monitor external {0} {1}'.format(self.want.name, command)
    uri = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs='-c "{0}"'.format(command))
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)
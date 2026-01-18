from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def provision_dedicated_on_device(self):
    params = self.want.api_params()
    uri = 'https://{0}:{1}/mgmt/tm/sys/provision/'.format(self.client.provider['server'], self.client.provider['server_port'])
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
    resources = [x['name'] for x in response['items'] if x['name'] != self.want.module]
    with TransactionContextManager(self.client) as transact:
        for resource in resources:
            target = uri + resource
            resp = transact.api.patch(target, json=dict(level='none'))
            try:
                response = resp.json()
            except ValueError as ex:
                raise F5ModuleError(str(ex))
            if 'code' in response and response['code'] in [400, 404]:
                if 'message' in response:
                    raise F5ModuleError(response['message'])
                else:
                    raise F5ModuleError(resp.content)
        target = uri + self.want.module
        resp = transact.api.patch(target, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if 'code' in response and response['code'] in [400, 404]:
            if 'message' in response:
                raise F5ModuleError(response['message'])
            else:
                raise F5ModuleError(resp.content)
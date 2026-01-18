from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ansible.module_utils._text import to_bytes
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def list_users_on_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/auth/user/'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = "?$filter=partition+eq+'{0}'".format(self.want.partition)
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    return response
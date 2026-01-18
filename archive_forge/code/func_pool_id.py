from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def pool_id(self):
    filter = "(name%20eq%20'{0}')".format(self.pool)
    uri = 'https://{0}:{1}/mgmt/cm/device/licensing/pool/regkey/licenses?$filter={2}&$top=1'.format(self.client.provider['server'], self.client.provider['server_port'], filter)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status == 200 and response['totalItems'] == 0:
        raise F5ModuleError('No pool with the specified name was found.')
    elif 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp._content)
    return response['items'][0]['id']
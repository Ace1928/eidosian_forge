from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def hotfix_exists(self):
    result = False
    uri = 'https://{0}:{1}/mgmt/tm/sys/software/hotfix/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    errors = [401, 403, 409, 500, 501, 502, 503, 504]
    if resp.status in errors or ('code' in response and response['code'] in errors):
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    if 'items' in response:
        for item in response['items']:
            if item['name'].startswith(self.want.filename):
                self._set_image_url(item)
                self.image_type = 'hotfix'
                result = True
                break
    return result
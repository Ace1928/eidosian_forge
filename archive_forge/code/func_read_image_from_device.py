from __future__ import absolute_import, division, print_function
import time
import ssl
from datetime import datetime
from ansible.module_utils.six.moves.urllib.error import URLError
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def read_image_from_device(self, type):
    uri = 'https://{0}:{1}/mgmt/tm/sys/software/{2}/'.format(self.client.provider['server'], self.client.provider['server_port'], type)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'items' in response:
        for item in response['items']:
            if item['name'].startswith(self.image):
                return item
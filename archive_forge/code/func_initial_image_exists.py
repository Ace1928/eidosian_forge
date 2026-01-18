from __future__ import absolute_import, division, print_function
import time
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def initial_image_exists(self, image):
    uri = 'https://{0}:{1}/mgmt/tm/sys/software/image/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError:
        return False
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False
    for resource in response['items']:
        if resource['name'].startswith(image):
            return True
    return False
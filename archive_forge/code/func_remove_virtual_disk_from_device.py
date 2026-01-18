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
def remove_virtual_disk_from_device(self):
    check = '{0}'.format(self.have.virtual_disk)
    response = self.get_virtual_disks_on_device()
    for resource in response['items']:
        if resource['name'].startswith(check):
            uri = 'https://{0}:{1}/mgmt/tm/vcmp/virtual-disk/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], resource['name'].replace('/', '~'))
            response = self.client.api.delete(uri)
            if response.status == 200:
                continue
            else:
                raise F5ModuleError(response.content)
    return True
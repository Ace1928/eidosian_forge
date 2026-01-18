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
@property
def block_device_image_info(self):
    if self._values['block_device_image_info']:
        block_device_image = self._values['block_device_image_info']
    else:
        block_device_image = self.read_block_device_image()
        self._values['block_device_image_info'] = block_device_image
    return block_device_image
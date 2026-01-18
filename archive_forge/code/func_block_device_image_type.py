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
def block_device_image_type(self):
    if self._values['block_device_image_type']:
        return self._values['block_device_image_type']
    if 'software:block-device-image' in self.block_device_image_info['kind']:
        self._values['block_device_image_type'] = 'block-device-image'
    else:
        self._values['block_device_image_type'] = 'block-device-hotfix'
    return self._values['block_device_image_type']
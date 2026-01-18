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
def image_type(self):
    if self._values['image_type']:
        return self._values['image_type']
    if 'software:image' in self.image_info['kind']:
        self._values['image_type'] = 'image'
    else:
        self._values['image_type'] = 'hotfix'
    return self._values['image_type']
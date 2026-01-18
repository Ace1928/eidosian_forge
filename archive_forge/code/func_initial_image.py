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
@property
def initial_image(self):
    if self._values['initial_image'] is None:
        return None
    if self.initial_image_exists(self._values['initial_image']):
        return self._values['initial_image']
    raise F5ModuleError("The specified 'initial_image' does not exist on the remote device.")
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
def allowed_slots(self):
    if self.want.allowed_slots is None:
        return None
    if self.have.allowed_slots is None:
        return self.want.allowed_slots
    if set(self.want.allowed_slots) != set(self.have.allowed_slots):
        return self.want.allowed_slots
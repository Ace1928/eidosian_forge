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
def mgmt_route(self):
    if self._values['mgmt_route'] is None:
        return None
    elif self._values['mgmt_route'] == 'none':
        return 'none'
    if is_valid_ip(self._values['mgmt_route']):
        return self._values['mgmt_route']
    else:
        raise F5ModuleError("The specified 'mgmt_route' is not a valid IP address.")
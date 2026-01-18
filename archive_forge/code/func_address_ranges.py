from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def address_ranges(self):
    if self.want.address_ranges is None:
        return None
    elif self.have.address_ranges is None:
        return self.want.address_ranges
    if sorted(self.want.address_ranges) != sorted(self.have.address_ranges):
        return self.want.address_ranges
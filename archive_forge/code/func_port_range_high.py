from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def port_range_high(self):
    if self._values['port_range_high'] is None:
        return None
    if 0 <= self._values['port_range_high'] <= 65535:
        return self._values['port_range_high']
    raise F5ModuleError("Valid 'port_range_high' must be in range 0 - 65535.")
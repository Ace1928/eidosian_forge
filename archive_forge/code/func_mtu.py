from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def mtu(self):
    if self._values['mtu'] is None:
        return None
    if int(self._values['mtu']) < 576 or int(self._values['mtu']) > 9198:
        raise F5ModuleError('The mtu value must be between 576 - 9198')
    return int(self._values['mtu'])
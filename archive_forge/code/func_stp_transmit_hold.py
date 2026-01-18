from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def stp_transmit_hold(self):
    if self._values['stp'] is None:
        return None
    if self._values['stp']['transmit_hold'] is None:
        return None
    if 1 <= self._values['stp']['transmit_hold'] <= 10:
        return self._values['stp']['transmit_hold']
    raise F5ModuleError("Valid 'transmit_hold' must be in range 1 - 10 packets.")
from __future__ import absolute_import, division, print_function
import copy
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def lldp_tlvmap(self):
    if self._values['lldp_tlvmap'] is None:
        return None
    if self._values['lldp_tlvmap'] == 0:
        return self._values['lldp_tlvmap']
    if 8 <= self._values['lldp_tlvmap'] <= 114680:
        return self._values['lldp_tlvmap']
    raise F5ModuleError('TLV value {0} is out of valid range of: 8 - 114680.')
from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def syn_rto_base(self):
    if self._values['syn_rto_base'] is None:
        return None
    if 0 <= self._values['syn_rto_base'] <= 5000:
        return self._values['syn_rto_base']
    raise F5ModuleError("Valid 'syn_rto_base' must be in range 0 - 5000 milliseconds.")
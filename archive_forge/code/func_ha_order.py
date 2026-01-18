from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ha_order(self):
    if self.want.ha_order is None:
        return None
    if self.have.ha_order is None and self.want.ha_order == 'none':
        return None
    if self.want.ha_order != self.have.ha_order:
        return self.want.ha_order
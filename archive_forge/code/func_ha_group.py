from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ha_group(self):
    if self.want.ha_group is None:
        return None
    if self.have.ha_group is None and self.want.ha_group == 'none':
        return None
    if self.want.ha_group != self.have.ha_group:
        if self.have.auto_failback == 'true' and self.want.auto_failback != 'false':
            raise F5ModuleError('The auto_failback parameter on the device must disabled to use ha_group failover method.')
        return self.want.ha_group
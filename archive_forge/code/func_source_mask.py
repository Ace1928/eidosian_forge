from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def source_mask(self):
    if self._values['source_mask'] is None:
        return None
    elif self._values['source_mask'] == 'any':
        return 0
    try:
        int(self._values['source_mask'])
        raise F5ModuleError("'source_mask' must not be in CIDR format.")
    except ValueError:
        pass
    if is_valid_ip(self._values['source_mask']):
        return self._values['source_mask']
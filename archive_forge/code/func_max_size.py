from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def max_size(self):
    if self._values['max_size'] is None:
        return None
    if 0 <= self._values['max_size'] <= 4294967295:
        return self._values['max_size']
    raise F5ModuleError("Valid 'max_size' must be in range 0 - 4294967295 bytes.")
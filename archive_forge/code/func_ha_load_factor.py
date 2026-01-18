from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def ha_load_factor(self):
    if self._values['ha_load_factor'] is None:
        return None
    value = self._values['ha_load_factor']
    if value < 1 or value > 1000:
        raise F5ModuleError('Invalid ha_load_factor value, correct range is 1 - 1000, specified value: {0}.'.format(value))
    return value
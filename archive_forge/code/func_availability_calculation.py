from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def availability_calculation(self):
    if self._values['availability_calculation'] is None:
        return None
    elif self._values['availability_calculation'] in ['any', 'when_any_available']:
        return 'any'
    elif self._values['availability_calculation'] in ['all', 'when_all_available']:
        return 'all'
    elif self._values['availability_calculation'] in ['none', 'always']:
        return 'none'
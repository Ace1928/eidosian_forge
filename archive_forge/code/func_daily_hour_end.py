from __future__ import absolute_import, division, print_function
import re
import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def daily_hour_end(self):
    if self._values['daily_hour_end'] is None:
        return None
    if self._values['daily_hour_start'] == 'all-day':
        return '24:00'
    if not self._values['daily_hour_end'] == '24:00':
        self._validate_time(self._values['daily_hour_end'])
    return self._values['daily_hour_end']
from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def rtbh_duration(self):
    if self._values['rtbh_duration'] is None:
        return None
    if 0 <= self._values['rtbh_duration'] <= 4294967295:
        return self._values['rtbh_duration']
    raise F5ModuleError("Valid 'rtbh_duration' must be in range 0 - 4294967295 seconds.")
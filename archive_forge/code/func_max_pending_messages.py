from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def max_pending_messages(self):
    if self._values['max_pending_messages'] is None:
        return None
    if 0 <= self._values['max_pending_messages'] <= 65535:
        return self._values['max_pending_messages']
    raise F5ModuleError("Valid 'max_pending_messages' must be in range 0 - 65535 messages.")
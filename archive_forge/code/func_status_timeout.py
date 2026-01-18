from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def status_timeout(self):
    divisor = 100
    timeout = self._values['status_timeout']
    if timeout < 150 or timeout > 3600:
        raise F5ModuleError('Timeout value must be between 150 and 3600 seconds.')
    delay = timeout / divisor
    return (delay, divisor)
from __future__ import absolute_import, division, print_function
import os
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def cache_timeout(self):
    if self._values['cache_timeout'] is None:
        return None
    if 0 <= self._values['cache_timeout'] <= 4194304:
        return self._values['cache_timeout']
    raise F5ModuleError("Valid 'cache_timeout' must be in range 0 - 86400 seconds.")
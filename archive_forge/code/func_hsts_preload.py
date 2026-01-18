from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def hsts_preload(self):
    if self._values['hsts_preload'] is None:
        return None
    elif self._values['hsts_preload'] == 'enabled':
        return 'yes'
    return 'no'
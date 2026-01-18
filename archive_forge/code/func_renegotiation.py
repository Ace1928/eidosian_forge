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
def renegotiation(self):
    result = flatten_boolean(self._values['renegotiation'])
    if result is None:
        return None
    if result == 'yes':
        return 'enabled'
    return 'disabled'
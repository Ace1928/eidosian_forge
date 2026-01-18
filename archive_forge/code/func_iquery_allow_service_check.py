from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def iquery_allow_service_check(self):
    if self._values['iquery_allow_service_check'] is None:
        return None
    elif self._values['iquery_allow_service_check']:
        return 'yes'
    return 'no'
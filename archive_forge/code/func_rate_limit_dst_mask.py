from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def rate_limit_dst_mask(self):
    if self._values['rate_limit_dst_mask'] is None:
        return None
    if 0 <= int(self._values['rate_limit_dst_mask']) <= 4294967295:
        return int(self._values['rate_limit_dst_mask'])
    raise F5ModuleError("Valid 'rate_limit_dst_mask' must be in range 0 - 4294967295.")
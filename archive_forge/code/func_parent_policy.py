from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def parent_policy(self):
    if self._values['parent_policy'] is None:
        return None
    result = self._values['parent_policy']['fullPath']
    return result
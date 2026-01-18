from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.urls import urlparse
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def package_root(self):
    if self._values['package'] is None:
        return None
    base = os.path.basename(self._values['package'])
    result = os.path.splitext(base)
    return result[0]
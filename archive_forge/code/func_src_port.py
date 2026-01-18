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
def src_port(self):
    if self._values['src_port'] is None:
        return None
    if 0 <= self._values['src_port'] <= 65535:
        return self._values['src_port']
    raise F5ModuleError("Valid 'src_port' must be in range 0 - 65535 inclusive.")
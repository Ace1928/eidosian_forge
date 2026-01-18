from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def pba_block_idle_timeout(self):
    if self._values['pba_block_idle_timeout'] is None:
        return None
    if 0 <= self._values['pba_block_idle_timeout'] <= 4294967295:
        return self._values['pba_block_idle_timeout']
    raise F5ModuleError("Valid 'pba_block_idle_timeout' must be in range 0 - 4294967295.")
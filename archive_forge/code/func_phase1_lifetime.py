from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def phase1_lifetime(self):
    if self._values['phase1_lifetime'] is None:
        return None
    if 1 <= int(self._values['phase1_lifetime']) <= 4294967295:
        return int(self._values['phase1_lifetime'])
    raise F5ModuleError("Valid 'phase1_lifetime' must be in range 1 - 4294967295.")
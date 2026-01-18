from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def persistence(self):
    to_filter = dict(mode=self._values['persistence_mode'], timeout=self._values['persistence_timeout'])
    result = self._filter_params(to_filter)
    if result:
        return result
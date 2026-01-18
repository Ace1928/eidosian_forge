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
def sflow(self):
    to_filter = dict(poll_interval=self._values['poll_interval'], poll_interval_global=self._values['poll_interval_global'], sampling_rate=self._values['sampling_rate'], sampling_rate_global=self._values['sampling_rate_global'])
    result = self._filter_params(to_filter)
    if result:
        return result
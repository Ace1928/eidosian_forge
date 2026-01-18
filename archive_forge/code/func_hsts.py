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
def hsts(self):
    result = dict()
    if self._values['hsts_mode'] is not None:
        result['mode'] = self._values['hsts_mode']
    if self._values['maximum_age'] is not None:
        result['maximumAge'] = self._values['maximum_age']
    if self._values['include_subdomains'] is not None:
        result['includeSubdomains'] = self._values['include_subdomains']
    if self._values['hsts_preload'] is not None:
        result['preload'] = self._values['hsts_preload']
    if not result:
        return None
    return result
from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def heavy_urls(self):
    tmp = dict()
    tmp['auto_detect'] = flatten_boolean(self._values['auto_detect'])
    tmp['latency_threshold'] = self._values['latency_threshold']
    tmp['exclude'] = self._values['hw_url_exclude']
    tmp['include'] = self._convert_include_list(self._values['hw_url_include'])
    result = self._filter_params(tmp)
    if result:
        return result
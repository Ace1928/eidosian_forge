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
def geolocations(self):
    tmp = dict()
    tmp['blacklist'] = self._values['geo_blacklist']
    tmp['whitelist'] = self._values['geo_whitelist']
    result = self._filter_params(tmp)
    if result:
        return result
from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def proxy_server_pool(self):
    if self._values['proxy_server_pool'] is None:
        return None
    result = fq_name(self.partition, self._values['proxy_server_pool'])
    return result
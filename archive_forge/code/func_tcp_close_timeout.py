from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def tcp_close_timeout(self):
    if self._values['tcp_close_timeout'] is None:
        return None
    try:
        return int(self._values['tcp_close_timeout'])
    except ValueError:
        return self._values['tcp_close_timeout']
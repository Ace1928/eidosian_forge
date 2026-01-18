from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def tcp_handshake_timeout(self):
    if self._values['tcp_handshake_timeout'] is None:
        return None
    try:
        return int(self._values['tcp_handshake_timeout'])
    except ValueError:
        if self._values['tcp_handshake_timeout'] in ['disabled', 'immediate']:
            return 0
        return self._values['tcp_handshake_timeout']
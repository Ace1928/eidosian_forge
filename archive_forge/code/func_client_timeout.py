from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def client_timeout(self):
    if self._values['client_timeout'] is None:
        return None
    try:
        return int(self._values['client_timeout'])
    except ValueError:
        if self._values['client_timeout'] == 0:
            return 'immediate'
        if self._values['client_timeout'] == 4294967295:
            return 'indefinite'
        return self._values['client_timeout']
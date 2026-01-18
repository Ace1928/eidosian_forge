from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def entry_timeout(self):
    if self._values['entry_timeout'] in [None, 'indefinite']:
        return self._values['entry_timeout']
    timeout = int(self._values['entry_timeout'])
    if 1 > timeout > 4294967295:
        raise F5ModuleError("'timeout' value must be between 1 and 4294967295, or the value 'indefinite'.")
    return timeout
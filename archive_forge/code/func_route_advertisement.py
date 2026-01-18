from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def route_advertisement(self):
    if self._values['route_advertisement'] is None:
        return None
    version = tmos_version(self.client)
    if Version(version) <= Version('13.0.0'):
        if self._values['route_advertisement'] == 'disabled':
            return 'disabled'
        else:
            return 'enabled'
    else:
        return self._values['route_advertisement']
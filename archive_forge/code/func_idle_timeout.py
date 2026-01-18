from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def idle_timeout(self):
    if self._values['idle_timeout'] is None:
        return None
    if self._values['idle_timeout'] == 'indefinite':
        return self._values['idle_timeout']
    if self._values['idle_timeout'] in ['0', 'immediate']:
        return 'immediate'
    return int(self._values['idle_timeout'])
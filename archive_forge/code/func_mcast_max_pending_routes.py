from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def mcast_max_pending_routes(self):
    if self._values['multicast'] is None:
        return None
    if self._values['multicast']['max_pending_routes'] is None:
        return None
    if 0 <= self._values['multicast']['max_pending_routes'] <= 4294967295:
        return self._values['multicast']['max_pending_routes']
    raise F5ModuleError("Valid 'max_pending_routes' must be in range 0 - 4294967295.")
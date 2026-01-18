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
def mcast_rate_limit(self):
    if self._values['multicast'] is None:
        return None
    result = flatten_boolean(self._values['multicast']['rate_limit'])
    if result == 'yes':
        return 'enabled'
    if result == 'no':
        return 'disabled'
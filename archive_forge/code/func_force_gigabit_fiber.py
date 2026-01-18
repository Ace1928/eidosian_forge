from __future__ import absolute_import, division, print_function
import copy
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def force_gigabit_fiber(self):
    result = flatten_boolean(self._values['force_gigabit_fiber'])
    if result == 'yes':
        return 'enabled'
    if result == 'no':
        return 'disabled'
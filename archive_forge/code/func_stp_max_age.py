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
def stp_max_age(self):
    if self._values['stp'] is None:
        return None
    if self._values['stp']['max_age'] is None:
        return None
    if 6 <= self._values['stp']['max_age'] <= 40:
        return self._values['stp']['max_age']
    raise F5ModuleError("Valid 'hello_time' must be in range 6 - 40 seconds.")
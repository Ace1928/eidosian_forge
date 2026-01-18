from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def process_recursion_desired(self):
    if self._values['process_recursion_desired'] is None:
        return None
    if self._values['process_recursion_desired']:
        return 'yes'
    return 'no'
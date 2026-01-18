from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def should_absent(self):
    result = self._absent_changed_options()
    if result:
        return True
    return False
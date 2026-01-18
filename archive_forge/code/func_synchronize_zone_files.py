from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def synchronize_zone_files(self):
    if self._values['synchronize_zone_files'] is None:
        return None
    elif self._values['synchronize_zone_files'] is False:
        return 'no'
    else:
        return 'yes'
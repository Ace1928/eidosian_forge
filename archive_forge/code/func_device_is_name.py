from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def device_is_name(self):
    if not self.device_is_address and (not self.device_is_id):
        return True
    return False
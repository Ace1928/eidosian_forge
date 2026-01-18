from __future__ import absolute_import, division, print_function
from_address:
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def smtp_server_password(self):
    if self.want.update_password == 'on_create':
        return None
    return self.want.smtp_server_password
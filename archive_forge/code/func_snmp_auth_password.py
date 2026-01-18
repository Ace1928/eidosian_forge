from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def snmp_auth_password(self):
    if self.want.update_password == 'always' and self.want.snmp_auth_password is not None:
        return self.want.snmp_auth_password
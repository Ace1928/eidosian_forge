from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def icmp_message(self):
    if self.want.icmp_message is None:
        return None
    if self.want.icmp_message is None and self.have.icmp_message is None:
        return None
    if self.have.icmp_message is None:
        return self.want.icmp_message
    if set(self.want.icmp_message) != set(self.have.icmp_message):
        return self.want.icmp_message
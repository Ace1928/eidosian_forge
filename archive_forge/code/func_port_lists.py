from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def port_lists(self):
    if self.want.port_lists is None:
        return None
    elif self.have.port_lists is None:
        return self.want.port_lists
    if sorted(self.want.port_lists) != sorted(self.have.port_lists):
        return self.want.port_lists
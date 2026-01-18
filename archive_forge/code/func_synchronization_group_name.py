from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def synchronization_group_name(self):
    if self.want.synchronization_group_name is None:
        return None
    if self.want.synchronization_group_name == '' and self.have.synchronization_group_name is None:
        return None
    if self.want.synchronization_group_name != self.have.synchronization_group_name:
        return self.want.synchronization_group_name
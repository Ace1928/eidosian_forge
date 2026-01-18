from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def name_servers(self):
    state = self.want.state
    if self.want.name_servers is None:
        return None
    if state == 'absent':
        if self.have.name_servers is None and self.want.name_servers:
            return None
        if set(self.want.name_servers) == set(self.have.name_servers):
            return []
        if set(self.want.name_servers) != set(self.have.name_servers):
            return list(set(self.want.name_servers).difference(self.have.name_servers))
    if not self.want.name_servers:
        if self.have.name_servers is None:
            return None
        if self.have.name_servers is not None:
            return self.want.name_servers
    if self.have.name_servers is None:
        return self.want.name_servers
    if set(self.want.name_servers) != set(self.have.name_servers):
        return self.want.name_servers
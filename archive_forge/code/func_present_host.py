from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_host(self):
    host = self.get_host()
    if not host:
        host = self._create_host(host)
    else:
        host = self._update_host(host)
    if host:
        host = self._handle_allocation_state(host)
    return host
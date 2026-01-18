from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_storage_pool(self):
    pool = self.get_storage_pool()
    if pool:
        pool = self._update_storage_pool(pool=pool)
    else:
        pool = self._create_storage_pool()
    if pool:
        pool = self._handle_allocation_state(pool=pool)
    return pool
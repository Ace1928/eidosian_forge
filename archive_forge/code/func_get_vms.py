from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_vms(self, entity):
    vms = self._connection.system_service().vms_service().list()
    resp = []
    for vm in vms:
        if vm.vm_pool is not None and vm.vm_pool.id == entity.id:
            resp.append(vm)
    return resp
from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def assigned_vms(self, affinity_group):
    if getattr(affinity_group.vms, 'href', None):
        return sorted([vm.id for vm in self._connection.follow_link(affinity_group.vms)])
    else:
        return sorted([vm.id for vm in affinity_group.vms])
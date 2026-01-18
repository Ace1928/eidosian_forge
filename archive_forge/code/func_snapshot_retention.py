from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def snapshot_retention(self, vm, vmid, retention):
    snapshots = self.snapshot(vm, vmid).get()[:-1]
    if retention > 0 and len(snapshots) > retention:
        for snap in sorted(snapshots, key=lambda x: x['snaptime'])[:len(snapshots) - retention]:
            self.snapshot(vm, vmid)(snap['name']).delete()
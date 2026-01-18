from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, list_snapshots, vmware_argument_spec
def get_snapshots_by_id_recursively(self, snapshots, snapid):
    snap_obj = []
    for snapshot in snapshots:
        if snapshot.id == snapid:
            snap_obj.append(snapshot)
        else:
            snap_obj = snap_obj + self.get_snapshots_by_id_recursively(snapshot.childSnapshotList, snapid)
    return snap_obj
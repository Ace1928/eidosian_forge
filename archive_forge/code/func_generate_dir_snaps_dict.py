from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_dir_snaps_dict(array):
    dir_snaps_info = {}
    snapshots = list(array.get_directory_snapshots().items)
    for snapshot in range(0, len(snapshots)):
        s_name = snapshots[snapshot].name
        dir_snaps_info[s_name] = {'destroyed': snapshots[snapshot].destroyed, 'source': snapshots[snapshot].source.name, 'suffix': snapshots[snapshot].suffix, 'client_name': snapshots[snapshot].client_name, 'snapshot_space': snapshots[snapshot].space.snapshots, 'total_physical_space': snapshots[snapshot].space.total_physical, 'unique_space': snapshots[snapshot].space.unique, 'used_provisioned': getattr(snapshots[snapshot].space, 'used_provisioned', None)}
        if LooseVersion(SUBS_API_VERSION) <= LooseVersion(array.get_rest_version()):
            dir_snaps_info[s_name]['total_used'] = snapshots[snapshot].space.total_used
        try:
            dir_snaps_info[s_name]['policy'] = snapshots[snapshot].policy.name
        except Exception:
            dir_snaps_info[s_name]['policy'] = ''
        if dir_snaps_info[s_name]['destroyed']:
            dir_snaps_info[s_name]['time_remaining'] = snapshots[snapshot].time_remaining
    return dir_snaps_info
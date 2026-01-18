from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_pgsnaps_dict(array):
    pgsnaps_info = {}
    snapshots = list(array.get_protection_group_snapshots().items)
    for snapshot in range(0, len(snapshots)):
        s_name = snapshots[snapshot].name
        pgsnaps_info[s_name] = {'destroyed': snapshots[snapshot].destroyed, 'source': snapshots[snapshot].source.name, 'suffix': snapshots[snapshot].suffix, 'snapshot_space': snapshots[snapshot].space.snapshots, 'used_provisioned': getattr(snapshots[snapshot].space, 'used_provisioned', None)}
        try:
            if pgsnaps_info[s_name]['destroyed']:
                pgsnaps_info[s_name]['time_remaining'] = snapshots[snapshot].time_remaining
        except AttributeError:
            pass
        try:
            pgsnaps_info[s_name]['manual_eradication'] = snapshots[snapshot].eradication_config.manual_eradication
        except AttributeError:
            pass
    return pgsnaps_info
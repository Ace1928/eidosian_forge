from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_snap_transfer_dict(blade):
    snap_transfer_info = {}
    snap_transfers = blade.file_system_snapshots.list_file_system_snapshots_transfer()
    for snap_transfer in range(0, len(snap_transfers.items)):
        transfer = snap_transfers.items[snap_transfer].name
        snap_transfer_info[transfer] = {'completed': snap_transfers.items[snap_transfer].completed, 'data_transferred': snap_transfers.items[snap_transfer].data_transferred, 'progress': snap_transfers.items[snap_transfer].progress, 'direction': snap_transfers.items[snap_transfer].direction, 'remote': snap_transfers.items[snap_transfer].remote.name, 'remote_snapshot': snap_transfers.items[snap_transfer].remote_snapshot.name, 'started': snap_transfers.items[snap_transfer].started, 'status': snap_transfers.items[snap_transfer].status}
    return snap_transfer_info
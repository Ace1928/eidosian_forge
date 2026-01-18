from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_vmsnap_dict(array):
    vmsnap_info = {}
    virt_snaps = list(array.get_virtual_machine_snapshots(vm_type='vvol').items)
    for snap in range(0, len(virt_snaps)):
        name = virt_snaps[snap].name
        vmsnap_info[name] = {'vm_type': virt_snaps[snap].vm_type, 'vm_id': virt_snaps[snap].vm_id, 'destroyed': virt_snaps[snap].destroyed, 'created': virt_snaps[snap].created, 'time_remaining': getattr(virt_snaps[snap], 'time_remaining', None), 'latest_pgsnapshot_name': getattr(virt_snaps[snap].recover_context, 'name', None), 'latest_pgsnapshot_id': getattr(virt_snaps[snap].recover_context, 'id', None)}
    return vmsnap_info
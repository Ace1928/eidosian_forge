from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_drives_dict(blade):
    """
    Drives information is only available for the Legend chassis.
    The Legend chassis product_name has // in it so only bother if
    that is the case.
    """
    drives_info = {}
    drives = list(blade.get_drives().items)
    if '//' in list(blade.get_arrays().items)[0].product_type:
        for drive in range(0, len(drives)):
            name = drives[drive].name
            drives_info[name] = {'progress': getattr(drives[drive], 'progress', None), 'raw_capacity': getattr(drives[drive], 'raw_capacity', None), 'status': getattr(drives[drive], 'status', None), 'details': getattr(drives[drive], 'details', None), 'type': getattr(drives[drive], 'type', None)}
    return drives_info
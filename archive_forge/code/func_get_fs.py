from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_fs(module, blade):
    """Return Filesystem or None"""
    fsys = []
    fsys.append(module.params['name'])
    try:
        res = blade.file_systems.list_file_systems(names=fsys)
        return res.items[0]
    except Exception:
        return None
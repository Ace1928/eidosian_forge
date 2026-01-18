from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_local_rl(module, blade):
    """Return Filesystem Replica Link or None"""
    try:
        res = blade.file_system_replica_links.list_file_system_replica_links(local_file_system_names=[module.params['name']])
        return res.items[0]
    except Exception:
        return None
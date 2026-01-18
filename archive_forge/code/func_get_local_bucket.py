from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def get_local_bucket(module, blade):
    """Return Bucket or None"""
    try:
        res = blade.buckets.list_buckets(names=[module.params['name']])
        return res.items[0]
    except Exception:
        return None
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def recover_bucket(module, blade):
    """Recover Bucket"""
    changed = True
    if not module.check_mode:
        try:
            api_version = blade.api_version.list_versions().versions
            if VERSIONING_VERSION in api_version:
                blade.buckets.update_buckets(names=[module.params['name']], bucket=BucketPatch(destroyed=False))
            else:
                blade.buckets.update_buckets(names=[module.params['name']], destroyed=Bucket(destroyed=False))
        except Exception:
            module.fail_json(msg='Object Store Bucket {0}: Recovery failed'.format(module.params['name']))
    module.exit_json(changed=changed)
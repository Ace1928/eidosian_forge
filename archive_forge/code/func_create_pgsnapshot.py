from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
def create_pgsnapshot(module, array):
    """Create Protection Group Snapshot"""
    api_version = array.get_rest_version()
    changed = True
    if not module.check_mode:
        suffix = ProtectionGroupSnapshot(suffix=module.params['suffix'])
        if LooseVersion(THROTTLE_API) >= LooseVersion(api_version):
            if list(array.get_protection_groups(names=[module.params['name']]).items)[0].target_count > 0:
                if module.params['now']:
                    res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], replicate_now=True, protection_group_snapshot=suffix)
                else:
                    res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], protection_group_snapshot=suffix, replicate=module.params['remote'])
            else:
                res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], protection_group_snapshot=suffix)
        elif list(array.get_protection_groups(names=[module.params['name']]).items)[0].target_count > 0:
            if module.params['now']:
                res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], replicate_now=True, allow_throttle=module.params['throttle'], protection_group_snapshot=suffix)
            else:
                res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], allow_throttle=module.params['throttle'], protection_group_snapshot=suffix, replicate=module.params['remote'])
        else:
            res = array.post_protection_group_snapshots(source_names=[module.params['name']], apply_retention=module.params['apply_retention'], allow_throttle=module.params['throttle'], protection_group_snapshot=suffix)
        if res.status_code != 200:
            module.fail_json(msg='Snapshot of pgroup {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
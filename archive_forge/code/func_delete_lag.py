from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_lag(module, blade):
    """Delete Link Aggregation Group"""
    changed = True
    if not module.check_mode:
        res = blade.delete_link_aggregation_groups(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete LAG {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
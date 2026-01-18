from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_snap_policy(module, blade):
    """Delete REST 2 snapshot policy

    If any rule parameters are provided then delete any rules that match
    all of the parameters provided.
    If no rule parameters are provided delete the entire policy
    """
    changed = False
    rule_delete = False
    if module.params['at'] or module.params['every'] or module.params['timezone'] or module.params['keep_for']:
        rule_delete = True
    if rule_delete:
        current_rules = list(blade.get_policies(names=[module.params['name']]).items)[0].rules
        for rule in range(0, len(current_rules)):
            current_rule = {'at': current_rules[rule].at, 'every': current_rules[rule].every, 'keep_for': current_rules[rule].keep_for, 'time_zone': current_rules[rule].time_zone}
            if not module.params['at']:
                delete_at = current_rules[rule].at
            else:
                delete_at = _convert_to_millisecs(module.params['at'])
            if module.params['keep_for']:
                delete_keep_for = module.params['keep_for']
            else:
                delete_keep_for = int(current_rules[rule].keep_for / 1000)
            if module.params['every']:
                delete_every = module.params['every']
            else:
                delete_every = int(current_rules[rule].every / 1000)
            if not module.params['timezone']:
                delete_tz = current_rules[rule].time_zone
            else:
                delete_tz = module.params['timezone']
            delete_rule = {'at': delete_at, 'every': delete_every * 1000, 'keep_for': delete_keep_for * 1000, 'time_zone': delete_tz}
            if current_rule == delete_rule:
                changed = True
                attr = PolicyPatch(remove_rules=[delete_rule])
                if not module.check_mode:
                    res = blade.patch_policies(destroy_snapshots=module.params['destroy_snapshots'], names=[module.params['name']], policy=attr)
                    if res.status_code != 200:
                        module.fail_json(msg='Failed to delete policy rule {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    else:
        changed = True
        if not module.check_mode:
            res = blade.delete_policies(names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to delete policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
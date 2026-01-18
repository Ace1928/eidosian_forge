from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_smb_share_policy(module, blade):
    """Create SMB Share Policy"""
    changed = True
    if not module.check_mode:
        res = blade.post_smb_share_policies(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to create SMB share policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if not module.params['enabled']:
            res = blade.patch_smb_share_policies(policy=SmbSharePolicy(enabled=False), names=[module.params['name']])
            if res.status_code != 200:
                blade.delete_smb_share_policies(names=[module.params['name']])
                module.fail_json(msg='Failed to create SMB share policy {0}.Error: {1}'.format(module.params['name'], res.errors[0].message))
        if not module.params['principal']:
            module.fail_json(msg='principal is required to create a new rule')
        else:
            rule = SmbSharePolicyRule(principal=module.params['principal'], change=module.params['change'], read=module.params['read'], full_control=module.params['full_control'])
            res = blade.post_smb_share_policies_rules(policy_names=[module.params['name']], rule=rule)
            if res.status_code != 200:
                module.fail_json(msg='Failed to create rule for policy {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
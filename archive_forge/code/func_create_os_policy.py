from __future__ import absolute_import, division, print_function
import os
import re
import platform
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_os_policy(module, blade):
    """Create Object Store Access Policy"""
    changed = True
    policy_name = module.params['account'] + '/' + module.params['name']
    versions = list(blade.get_versions().items)
    if not module.check_mode:
        res = blade.post_object_store_access_policies(names=[policy_name], policy=ObjectStoreAccessPolicyPost(description=module.params['desc']))
        if res.status_code != 200:
            module.fail_json(msg='Failed to create access policy {0}.'.format(policy_name))
        if module.params['rule']:
            if not module.params['actions'] or not module.params['object_resources']:
                module.fail_json(msg='Parameters `actions` and `object_resources` are required to create a new rule')
            conditions = PolicyRuleObjectAccessCondition(source_ips=module.params['source_ips'], s3_delimiters=module.params['s3_delimiters'], s3_prefixes=module.params['s3_prefixes'])
            if SMB_ENCRYPT_API_VERSION in versions:
                rule = PolicyRuleObjectAccessPost(actions=module.params['actions'], resources=module.params['object_resources'], conditions=conditions, effect=module.params['effect'])
            else:
                rule = PolicyRuleObjectAccessPost(actions=module.params['actions'], resources=module.params['object_resources'], conditions=conditions)
            res = blade.post_object_store_access_policies_rules(policy_names=policy_name, names=[module.params['rule']], enforce_action_restrictions=module.params['ignore_enforcement'], rule=rule)
            if res.status_code != 200:
                module.fail_json(msg='Failed to create rule {0} to policy {1}. Error: {2}'.format(module.params['rule'], policy_name, res.errors[0].message))
        if module.params['user']:
            member_name = module.params['account'] + '/' + module.params['user']
            res = blade.post_object_store_access_policies_object_store_users(member_names=[member_name], policy_names=[policy_name])
            if res.status_code != 200:
                module.fail_json(msg='Failed to add users to policy {0}. Error: {1} - {2}'.format(policy_name, res.errors[0].context, res.errors[0].message))
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def list_policy(module, blade):
    """List Object Store User Access Policies"""
    changed = True
    policy_list = []
    if not module.check_mode:
        if module.params['account'] and module.params['name']:
            username = module.params['account'] + '/' + module.params['name']
            user_policies = list(blade.get_object_store_access_policies_object_store_users(member_names=[username]).items)
            for user_policy in range(0, len(user_policies)):
                policy_list.append(user_policies[user_policy].policy.name)
        else:
            policies = blade.get_object_store_access_policies()
            p_list = list(policies.items)
            if policies.status_code != 200:
                module.fail_json(msg='Failed to get Object Store User Access Policies')
            for policy in range(0, len(p_list)):
                policy_list.append(p_list[policy].name)
    module.exit_json(changed=changed, policy_list=policy_list)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_s3user(module, blade):
    """Create Object Store Account"""
    s3user_facts = {}
    changed = True
    if not module.check_mode:
        user = module.params['account'] + '/' + module.params['name']
        blade.object_store_users.create_object_store_users(names=[user])
        if module.params['access_key'] and module.params['imported_key']:
            module.warn("'access_key: true' overrides imported keys")
        if module.params['access_key']:
            try:
                result = blade.object_store_access_keys.create_object_store_access_keys(object_store_access_key=ObjectStoreAccessKey(user={'name': user}))
                s3user_facts['fb_s3user'] = {'user': user, 'access_key': result.items[0].secret_access_key, 'access_id': result.items[0].name}
            except Exception:
                delete_s3user(module, blade, True)
                module.fail_json(msg='Object Store User {0}: Creation failed'.format(user))
        elif module.params['imported_key']:
            versions = blade.api_version.list_versions().versions
            if IMPORT_KEY_API_VERSION in versions:
                try:
                    blade.object_store_access_keys.create_object_store_access_keys(names=[module.params['imported_key']], object_store_access_key=ObjectStoreAccessKeyPost(user={'name': user}, secret_access_key=module.params['imported_secret']))
                except Exception:
                    delete_s3user(module, blade)
                    module.fail_json(msg='Object Store User {0}: Creation failed with imported access key'.format(user))
        if module.params['policy']:
            blade = get_system(module)
            api_version = list(blade.get_versions().items)
            if POLICY_API_VERSION in api_version:
                policy_list = module.params['policy']
                for policy in range(0, len(policy_list)):
                    if blade.get_object_store_access_policies(names=[policy_list[policy]]).status_code != 200:
                        module.warn('Policy {0} is not valid. Ignoring...'.format(policy_list[policy]))
                        policy_list.remove(policy_list[policy])
                username = module.params['account'] + '/' + module.params['name']
                for policy in range(0, len(policy_list)):
                    if not blade.get_object_store_users_object_store_access_policies(member_names=[username], policy_names=[policy_list[policy]]).items:
                        res = blade.post_object_store_access_policies_object_store_users(member_names=[username], policy_names=[policy_list[policy]])
                        if res.status_code != 200:
                            module.warn('Failed to add policy {0} to account user {1}. Skipping...'.format(policy_list[policy], username))
                if 'pure:policy/full-access' not in policy_list:
                    blade.delete_object_store_access_policies_object_store_users(member_names=[username], policy_names=['pure:policy/full-access'])
            else:
                module.warn('FlashBlade REST version not supported for user access policies. Skipping...')
    module.exit_json(changed=changed, s3user_info=s3user_facts)
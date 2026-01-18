from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_account(module, blade):
    """Create Active Directory Account"""
    changed = True
    if not module.params['existing']:
        ad_config = ActiveDirectoryPost(computer_name=module.params['computer'], directory_servers=module.params['directory_servers'], kerberos_servers=module.params['kerberos_servers'], domain=module.params['domain'], encryption_types=module.params['encryption'], fqdns=module.params['service_principals'], join_ou=module.params['join_ou'], user=module.params['username'], password=module.params['password'])
        if not module.check_mode:
            res = blade.post_active_directory(names=[module.params['name']], active_directory=ad_config)
            if res.status_code != 200:
                module.fail_json(msg='Failed to add Active Directory Account {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    else:
        ad_config = ActiveDirectoryPost(computer_name=module.params['computer'], directory_servers=module.params['directory_servers'], kerberos_servers=module.params['kerberos_servers'], domain=module.params['domain'], encryption_types=module.params['encryption'], user=module.params['username'], password=module.params['password'])
        if not module.check_mode:
            res = blade.post_active_directory(names=[module.params['name']], active_directory=ad_config, join_existing_account=True)
            if res.status_code != 200:
                module.fail_json(msg='Failed to add Active Directory Account {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)
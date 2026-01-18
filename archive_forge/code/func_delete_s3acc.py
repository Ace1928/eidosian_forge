from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_s3acc(module, blade):
    """Delete Object Store Account"""
    changed = True
    if not module.check_mode:
        count = len(blade.object_store_users.list_object_store_users(filter="name='" + module.params['name'] + "/*'").items)
        if count != 0:
            module.fail_json(msg='Remove all Users from Object Store Account {0}                                  before deletion'.format(module.params['name']))
        else:
            try:
                blade.object_store_accounts.delete_object_store_accounts(names=[module.params['name']])
            except Exception:
                module.fail_json(msg='Object Store Account {0}: Deletion failed'.format(module.params['name']))
    module.exit_json(changed=changed)
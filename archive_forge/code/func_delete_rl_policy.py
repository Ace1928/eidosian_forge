from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_rl_policy(module, blade):
    """Delete Policy from Filesystem Replica Link"""
    changed = True
    if not module.check_mode:
        current_policy = blade.file_system_replica_links.list_file_system_replica_link_policies(local_file_system_names=[module.params['name']], policy_names=[module.params['policy']])
        if current_policy.items:
            try:
                blade.file_system_replica_links.delete_file_system_replica_link_policies(policy_names=[module.params['policy']], local_file_system_names=[module.params['name']], remote_names=[current_policy.items[0].link.remote.name])
            except Exception:
                module.fail_json(msg='Failed to remove policy {0} from replica link {1}.'.format(module.params['policy'], module.params['name']))
        else:
            changed = False
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def add_rl_policy(module, blade):
    """Add Policy to Filesystem Replica Link"""
    changed = False
    if not module.params['target_array']:
        module.params['target_array'] = blade.file_system_replica_links.list_file_system_replica_links(local_file_system_names=[module.params['name']]).items[0].remote.name
    remote_array = _check_connected(module, blade)
    try:
        already_a_policy = blade.file_system_replica_links.list_file_system_replica_link_policies(local_file_system_names=[module.params['name']], policy_names=[module.params['policy']], remote_names=[remote_array.remote.name])
        if not already_a_policy.items:
            changed = True
            if not module.check_mode:
                blade.file_system_replica_links.create_file_system_replica_link_policies(policy_names=[module.params['policy']], local_file_system_names=[module.params['name']], remote_names=[remote_array.remote.name])
    except Exception:
        module.fail_json(msg='Failed to add policy {0} to replica link {1}.'.format(module.params['policy'], module.params['name']))
    module.exit_json(changed=changed)
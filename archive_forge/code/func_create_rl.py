from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def create_rl(module, blade):
    """Create Filesystem Replica Link"""
    changed = True
    if not module.check_mode:
        try:
            remote_array = _check_connected(module, blade)
            if remote_array:
                if not module.params['target_fs']:
                    module.params['target_fs'] = module.params['name']
                if not module.params['policy']:
                    blade.file_system_replica_links.create_file_system_replica_links(local_file_system_names=[module.params['name']], remote_file_system_names=[module.params['target_fs']], remote_names=[remote_array.remote.name])
                else:
                    blade.file_system_replica_links.create_file_system_replica_links(local_file_system_names=[module.params['name']], remote_file_system_names=[module.params['target_fs']], remote_names=[remote_array.remote.name], file_system_replica_link=FileSystemReplicaLink(policies=[LocationReference(name=module.params['policy'])]))
            else:
                module.fail_json(msg='Target array {0} is not connected'.format(module.params['target_array']))
        except Exception:
            module.fail_json(msg='Failed to create filesystem replica link for {0}.'.format(module.params['name']))
    module.exit_json(changed=changed)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, human_to_bytes
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def delete_fs(module, blade):
    """Delete Filesystem"""
    changed = True
    if not module.check_mode:
        try:
            api_version = blade.api_version.list_versions().versions
            if REPLICATION_API_VERSION in api_version:
                if NFSV4_API_VERSION in api_version:
                    blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(v3_enabled=False, v4_1_enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True), delete_link_on_eradication=module.params['delete_link'])
                else:
                    blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True), delete_link_on_eradication=module.params['delete_link'])
            elif NFSV4_API_VERSION in api_version:
                blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(v3_enabled=False, v4_1_enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True))
            else:
                blade.file_systems.update_file_systems(name=module.params['name'], attributes=FileSystem(nfs=NfsRule(enabled=False), smb=ProtocolRule(enabled=False), http=ProtocolRule(enabled=False), destroyed=True))
            if module.params['eradicate']:
                try:
                    blade.file_systems.delete_file_systems(name=module.params['name'])
                except Exception:
                    module.fail_json(msg='Failed to delete filesystem {0}.'.format(module.params['name']))
        except Exception:
            module.fail_json(msg='Failed to update filesystem {0} prior to deletion.'.format(module.params['name']))
    module.exit_json(changed=changed)
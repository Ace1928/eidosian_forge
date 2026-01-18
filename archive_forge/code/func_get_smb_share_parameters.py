from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_smb_share_parameters():
    """
    This method provides parameters required for the ansible smb share
    modules on Unity
    """
    return dict(share_name=dict(), share_id=dict(), filesystem_name=dict(), filesystem_id=dict(), snapshot_name=dict(), snapshot_id=dict(), nas_server_name=dict(), nas_server_id=dict(), path=dict(no_log=True), umask=dict(), description=dict(), offline_availability=dict(choices=['MANUAL', 'DOCUMENTS', 'PROGRAMS', 'NONE']), is_abe_enabled=dict(type='bool'), is_branch_cache_enabled=dict(type='bool'), is_continuous_availability_enabled=dict(type='bool'), is_encryption_enabled=dict(type='bool'), state=dict(required=True, choices=['present', 'absent'], type='str'))
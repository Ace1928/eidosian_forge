from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_smb_share(self, share_name, path, filesystem_obj=None, snapshot_obj=None, description=None, is_abe_enabled=None, is_branch_cache_enabled=None, is_continuous_availability_enabled=None, is_encryption_enabled=None, offline_availability=None, umask=None):
    """
        Create SMB Share
        :return: SMB Share Object if successful, else error.
        """
    if path is None or path == '':
        self.module.fail_json(msg='Please enter a valid path. Empty string or None provided.')
    if not filesystem_obj and (not snapshot_obj):
        self.module.fail_json(msg="Either Filesystem or Snapshot Resource's Name/ID is required to Create a SMB share")
    try:
        if filesystem_obj:
            return self.smb_share_conn_obj.create(cli=self.unity_conn._cli, name=share_name, fs=filesystem_obj, path=path, is_encryption_enabled=is_encryption_enabled, is_con_avail_enabled=is_continuous_availability_enabled, is_abe_enabled=is_abe_enabled, is_branch_cache_enabled=is_branch_cache_enabled, umask=umask, description=description, offline_availability=offline_availability)
        else:
            return self.smb_share_conn_obj.create_from_snap(cli=self.unity_conn._cli, name=share_name, snap=snapshot_obj, path=path, is_encryption_enabled=is_encryption_enabled, is_con_avail_enabled=is_continuous_availability_enabled, is_abe_enabled=is_abe_enabled, is_branch_cache_enabled=is_branch_cache_enabled, umask=umask, description=description, offline_availability=offline_availability)
    except Exception as e:
        error_msg = 'Failed to create SMB share %s with error %s' % (share_name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def delete_smb_share(self, smb_share_obj):
    """
        Delete SMB share if exists, else thrown error.
        """
    try:
        smb_share_obj.delete()
    except Exception as e:
        error_msg = 'Failed to Delete SMB share %s with error %s' % (smb_share_obj.name, str(e))
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
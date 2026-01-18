from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_nfs_share(self):
    """ Create nfs share from either filesystem/snapshot

        :return: nfs_share object
        :rtype: UnityNfsShare
        """
    if self.is_given_nfs_for_fs:
        return self.create_nfs_share_from_filesystem()
    elif self.is_given_nfs_for_fs is False:
        return self.create_nfs_share_from_snapshot()
    else:
        msg = 'Please provide filesystem or filesystem snapshot to create NFS export'
        LOG.error(msg)
        self.module.fail_json(msg=msg)
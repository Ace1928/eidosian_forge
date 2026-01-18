from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def delete_nfs_share(self, nfs_obj):
    """ Delete nfs share

        :param nfs: NFS share obj
        :type nfs: UnityNfsShare
        :return: None
        """
    try:
        LOG.info('Deleting nfs share: %s', self.get_nfs_id_or_name())
        nfs_obj.delete()
        LOG.info('Deleted nfs share')
    except Exception as e:
        msg = 'Failed to delete nfs share, error: %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
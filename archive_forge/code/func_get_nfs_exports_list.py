from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_exports_list(self):
    """Get the list of NFS exports on a given Unity storage system"""
    try:
        LOG.info('Getting NFS exports list')
        nfs_exports = self.unity.get_nfs_share()
        return result_list(nfs_exports)
    except Exception as e:
        msg = 'Get NFS exports from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_server_list(self):
    """Get the list of NFS servers on a given Unity storage system"""
    try:
        LOG.info('Getting NFS servers list')
        nfs_servers = self.unity.get_nfs_server()
        return nfs_server_result_list(nfs_servers)
    except Exception as e:
        msg = 'Get NFS servers list from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
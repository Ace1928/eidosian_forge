from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def delete_nfs_server(self, nfs_server_id, skip_unjoin=None, domain_username=None, domain_password=None):
    """Delete NFS server.
            :param: nfs_server_id: The ID of the NFS server
            :param: skip_unjoin: Flag indicating whether to unjoin SMB server account from AD before deletion
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :return: Return True if NFS server is deleted
        """
    LOG.info('Deleting NFS server')
    try:
        if not self.module.check_mode:
            nfs_obj = self.get_nfs_server_instance(nfs_server_id=nfs_server_id)
            nfs_obj.delete(skip_kdc_unjoin=skip_unjoin, username=domain_username, password=domain_password)
        return True
    except Exception as e:
        msg = 'Failed to delete NFS server: %s with error: %s' % (nfs_server_id, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)
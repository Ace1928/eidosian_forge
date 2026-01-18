from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_server_details(self, nfs_server_id=None, nas_server_id=None):
    """Get NFS server details.
            :param: nfs_server_id: The ID of the NFS server
            :param: nas_server_id: The name of the NAS server
            :return: Dict containing NFS server details if exists
        """
    LOG.info('Getting NFS server details')
    try:
        if nfs_server_id:
            nfs_server_details = self.unity_conn.get_nfs_server(_id=nfs_server_id)
            return nfs_server_details._get_properties()
        elif nas_server_id:
            nfs_server_details = self.unity_conn.get_nfs_server(nas_server=nas_server_id)
            if len(nfs_server_details) > 0:
                return process_dict(nfs_server_details._get_properties())
            return None
    except utils.HttpError as e:
        if e.http_status == 401:
            msg = 'Incorrect username or password provided.'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        else:
            err_msg = 'Failed to get details of NFS Server with error {0}'.format(str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
    except utils.UnityResourceNotFoundError as e:
        err_msg = 'Failed to get details of NFS Server with error {0}'.format(str(e))
        LOG.error(err_msg)
        return None
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
def modify_storage_pool(self, pool_id, modify_dict):
    """
        Modify the parameters of the storage pool.
        :param modify_dict: Dict containing parameters which are to be
         modified
        :param pool_id: Id of the pool.
        :return: True, if the operation is successful.
        """
    try:
        if 'new_name' in modify_dict:
            self.powerflex_conn.storage_pool.rename(pool_id, modify_dict['new_name'])
        if 'use_rmcache' in modify_dict:
            self.powerflex_conn.storage_pool.set_use_rmcache(pool_id, modify_dict['use_rmcache'])
        if 'use_rfcache' in modify_dict:
            self.powerflex_conn.storage_pool.set_use_rfcache(pool_id, modify_dict['use_rfcache'])
        if 'media_type' in modify_dict:
            self.powerflex_conn.storage_pool.set_media_type(pool_id, modify_dict['media_type'])
        return True
    except Exception as e:
        err_msg = 'Failed to update the storage pool {0} with error {1}'.format(pool_id, str(e))
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)
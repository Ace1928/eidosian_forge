from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def delete_tree_quota(self, tree_quota_id):
    """
        Delete quota tree of a filesystem.
        :param tree_quota_id: ID of quota tree
        :return: Boolean whether quota tree is deleted
        """
    try:
        delete_tree_quota_obj = self.unity_conn.delete_tree_quota(tree_quota_id=tree_quota_id)
        if delete_tree_quota_obj:
            return True
    except Exception as e:
        errormsg = 'Delete operation of quota tree id:{0} failed with error {1}'.format(tree_quota_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
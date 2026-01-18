from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def check_quota_tree_is_present(self, fs_id, path, tree_quota_id):
    """
            Check if quota tree is present in filesystem.
            :param fs_id: ID of filesystem where quota tree is searched.
            :param path: Path to the quota tree
            :param tree_quota_id: ID of the quota tree
            :return: ID of quota tree if it exists else None.
        """
    if tree_quota_id is None and path is None:
        return None
    all_tree_quota = self.unity_conn.get_tree_quota(filesystem=fs_id, id=tree_quota_id, path=path)
    if tree_quota_id and len(all_tree_quota) == 0 and (self.module.params['state'] == 'present'):
        errormsg = 'Tree quota %s does not exist.' % tree_quota_id
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if len(all_tree_quota) > 0:
        msg = 'Quota tree with id %s is present in filesystem %s' % (all_tree_quota[0].id, fs_id)
        LOG.info(msg)
        return all_tree_quota[0].id
    else:
        return None
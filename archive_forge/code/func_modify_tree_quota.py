from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_tree_quota(self, tree_quota_id, soft_limit, hard_limit, unit, description):
    """
            Modify quota tree of filesystem.
            :param tree_quota_id: ID of the quota tree
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :param description: Description of quota tree
            :return: Boolean value whether modify quota tree operation is successful.
        """
    try:
        if soft_limit is None and hard_limit is None:
            return False
        tree_quota_obj = self.unity_conn.get_tree_quota(tree_quota_id)._get_properties()
        if soft_limit is None:
            soft_limit_in_bytes = tree_quota_obj['soft_limit']
        else:
            soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
        if hard_limit is None:
            hard_limit_in_bytes = tree_quota_obj['hard_limit']
        else:
            hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
        if description is None:
            description = tree_quota_obj['description']
        if tree_quota_obj:
            if tree_quota_obj['soft_limit'] == soft_limit_in_bytes and tree_quota_obj['hard_limit'] == hard_limit_in_bytes and (tree_quota_obj['description'] == description):
                return False
            else:
                modify_tree_quota = self.unity_conn.modify_tree_quota(tree_quota_id=tree_quota_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes, description=description)
                LOG.info('Successfully modified quota tree')
                if modify_tree_quota:
                    return True
    except Exception as e:
        errormsg = 'Modify quota tree operation {0} failed with error {1}'.format(tree_quota_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
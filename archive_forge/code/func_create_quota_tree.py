from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def create_quota_tree(self, fs_id, soft_limit, hard_limit, unit, path, description):
    """
            Create quota tree of a filesystem.
            :param fs_id: ID of filesystem where quota tree is to be created.
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :param path: Path to quota tree
            :param description: Description for quota tree
            :return: Dict containing new quota tree details.
        """
    if soft_limit is None and hard_limit is None:
        errormsg = 'Both soft limit and hard limit cannot be empty. Please provide atleast one to create quota tree.'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
    hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
    try:
        obj_tree_quota = self.unity_conn.create_tree_quota(filesystem_id=fs_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes, path=path, description=description)
        LOG.info('Successfully created quota tree')
        if obj_tree_quota:
            return obj_tree_quota
        else:
            return None
    except Exception as e:
        errormsg = 'Create quota tree operation at path {0} failed in filesystem {1} with error {2}'.format(path, fs_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
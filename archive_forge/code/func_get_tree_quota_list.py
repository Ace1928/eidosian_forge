from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_tree_quota_list(self):
    """Get the list of quota trees on a given Unity storage system"""
    try:
        LOG.info('Getting quota tree list')
        tree_quotas = self.unity.get_tree_quota()
        return tree_quota_result_list(tree_quotas)
    except Exception as e:
        msg = 'Get quota trees from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
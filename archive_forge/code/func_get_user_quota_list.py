from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_user_quota_list(self):
    """Get the list of user quotas on a given Unity storage system"""
    try:
        LOG.info('Getting user quota list')
        user_quotas = self.unity.get_user_quota()
        return user_quota_result_list(user_quotas)
    except Exception as e:
        msg = 'Get user quotas from unity array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)
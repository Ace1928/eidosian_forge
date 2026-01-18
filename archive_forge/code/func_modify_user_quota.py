from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_user_quota(self, user_quota_id, soft_limit, hard_limit, unit):
    """
            Modify user quota of filesystem by its uid/username/user quota id.
            :param user_quota_id: ID of the user quota
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :return: Boolean value whether modify user quota operation is successful.
        """
    if soft_limit is None and hard_limit is None:
        return False
    user_quota_obj = self.unity_conn.get_user_quota(user_quota_id)._get_properties()
    if soft_limit is None:
        soft_limit_in_bytes = user_quota_obj['soft_limit']
    else:
        soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
    if hard_limit is None:
        hard_limit_in_bytes = user_quota_obj['hard_limit']
    else:
        hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
    if user_quota_obj:
        if user_quota_obj['soft_limit'] == soft_limit_in_bytes and user_quota_obj['hard_limit'] == hard_limit_in_bytes:
            return False
    else:
        error_msg = 'The user quota does not exist.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    try:
        obj_user_quota = self.unity_conn.modify_user_quota(user_quota_id=user_quota_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes)
        LOG.info('Successfully modified user quota')
        if obj_user_quota:
            return True
    except Exception as e:
        errormsg = 'Modify user quota {0} failed with error {1}'.format(user_quota_id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
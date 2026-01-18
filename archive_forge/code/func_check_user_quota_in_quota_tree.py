from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def check_user_quota_in_quota_tree(self, tree_quota_id, uid, unix, win_name, user_quota_id):
    """
            Check if user quota is present in quota tree.
            :param tree_quota_id: ID of quota tree where user quota is searched.
            :param uid: UID of user quota
            :param unix: Unix name of user quota
            :param win_name: Windows name of user quota
            :param user_quota_id: ID of the user quota
            :return: ID of user quota if it exists in quota tree else None.
        """
    if not self.check_user_type_provided(win_name, uid, unix):
        return None
    user_quota_name = uid if uid else unix if unix else win_name if win_name else user_quota_id
    user_quota_obj = self.unity_conn.get_user_quota(tree_quota=tree_quota_id, uid=uid, windows_names=win_name, unix_name=unix, id=user_quota_id)
    if len(user_quota_obj) > 0:
        msg = 'User quota %s is present in quota tree %s ' % (user_quota_name, tree_quota_id)
        LOG.info(msg)
        return user_quota_obj[0].id
    else:
        return None
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def check_user_type_provided(self, win_name, uid, unix_name):
    """Checks if user type or uid is provided
           :param win_name: Windows name of user quota
           :param uid: UID of user quota
           :param unix_name: Unix name of user quota"""
    if win_name is None and uid is None and (unix_name is None):
        return False
    else:
        return True
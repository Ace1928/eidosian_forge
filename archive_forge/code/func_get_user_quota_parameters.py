from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_user_quota_parameters():
    """This method provide parameters required for the ansible filesystem
       user quota module on Unity"""
    return dict(filesystem_id=dict(required=False, type='str'), filesystem_name=dict(required=False, type='str'), state=dict(required=True, type='str', choices=['present', 'absent']), user_type=dict(required=False, type='str', choices=['Windows', 'Unix']), user_name=dict(required=False, type='str'), uid=dict(required=False, type='str'), win_domain=dict(required=False, type='str'), hard_limit=dict(required=False, type='int'), soft_limit=dict(required=False, type='int'), cap_unit=dict(required=False, type='str', choices=['MB', 'GB', 'TB']), user_quota_id=dict(required=False, type='str'), nas_server_name=dict(required=False, type='str'), nas_server_id=dict(required=False, type='str'), tree_quota_id=dict(required=False, type='str'), path=dict(required=False, type='str', no_log=True))
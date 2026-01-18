from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_quota_tree_parameters():
    """This method provide parameters required for the ansible
       quota tree module on Unity"""
    return dict(filesystem_id=dict(required=False, type='str'), filesystem_name=dict(required=False, type='str'), state=dict(required=True, type='str', choices=['present', 'absent']), hard_limit=dict(required=False, type='int'), soft_limit=dict(required=False, type='int'), cap_unit=dict(required=False, type='str', choices=['MB', 'GB', 'TB']), tree_quota_id=dict(required=False, type='str'), nas_server_name=dict(required=False, type='str'), nas_server_id=dict(required=False, type='str'), path=dict(required=False, type='str', no_log=True), description=dict(required=False, type='str'))
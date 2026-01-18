from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def compare_protected_branch(self, name, merge_access_levels, push_access_level):
    configured_merge = self.ACCESS_LEVEL[merge_access_levels]
    configured_push = self.ACCESS_LEVEL[push_access_level]
    current = self.protected_branch_exist(name=name)
    current_merge = current.merge_access_levels[0]['access_level']
    current_push = current.push_access_levels[0]['access_level']
    if current:
        if current.name == name and current_merge == configured_merge and (current_push == configured_push):
            return True
    return False
from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def member_as_good_access_level(self, group, user_id, access_level):
    member = self.find_member(group, user_id)
    return member.access_level == access_level
from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def get_member_in_a_group(self, gitlab_group_id, gitlab_user_id):
    member = None
    group = self._gitlab.groups.get(gitlab_group_id)
    try:
        member = group.members.get(gitlab_user_id)
        if member:
            return member
    except gitlab.exceptions.GitlabGetError as e:
        return None
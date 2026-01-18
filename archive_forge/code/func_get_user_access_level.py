from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def get_user_access_level(self, members, gitlab_user_id):
    for member in members:
        if member.id == gitlab_user_id:
            return member.access_level
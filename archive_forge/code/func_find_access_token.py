from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def find_access_token(self, group, name):
    access_tokens = group.access_tokens.list(all=True)
    for access_token in access_tokens:
        if access_token.name == name:
            self.access_token_object = access_token
            return False
    return False
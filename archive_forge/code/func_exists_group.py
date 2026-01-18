from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def exists_group(self, project_identifier):
    group = find_group(self._gitlab, project_identifier)
    if group:
        self.group_object = group
        return True
    return False
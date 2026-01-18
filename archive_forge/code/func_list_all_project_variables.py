from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def list_all_project_variables(self):
    return list(self.project.variables.list(**list_all_kwargs))
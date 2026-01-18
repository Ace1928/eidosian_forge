from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def delete_protected_branch(self, name):
    if self._module.check_mode:
        return True
    return self.project.protectedbranches.delete(name)
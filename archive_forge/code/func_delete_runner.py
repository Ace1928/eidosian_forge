from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def delete_runner(self):
    if self._module.check_mode:
        return True
    runner = self.runner_object
    return runner.delete()
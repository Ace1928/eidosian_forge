from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def create_runner(self, arguments):
    if self._module.check_mode:
        return True
    try:
        if arguments.get('token') is not None:
            runner = self._gitlab.runners.create(arguments)
        elif LooseVersion(gitlab.__version__) < LooseVersion('4.0.0'):
            self._module.fail_json(msg='New runner creation workflow requires python-gitlab 4.0.0 or higher')
        else:
            runner = self._gitlab.user.runners.create(arguments)
    except gitlab.exceptions.GitlabCreateError as e:
        self._module.fail_json(msg='Failed to create runner: %s ' % to_native(e))
    return runner
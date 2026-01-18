from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def add_ssh_key_to_user(self, user, sshkey):
    if not self.ssh_key_exists(user, sshkey['name']):
        if self._module.check_mode:
            return True
        try:
            parameter = {'title': sshkey['name'], 'key': sshkey['file']}
            if sshkey['expires_at'] is not None:
                parameter['expires_at'] = sshkey['expires_at']
            user.keys.create(parameter)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to assign sshkey to user: %s' % to_native(e))
        return True
    return False
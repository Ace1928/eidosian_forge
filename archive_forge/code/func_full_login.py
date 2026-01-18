from __future__ import (absolute_import, division, print_function)
import errno
import json
import os
import re
from subprocess import Popen, PIPE
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def full_login(self):
    if self.auto_login is not None:
        if None in [self.auto_login.get('subdomain'), self.auto_login.get('username'), self.auto_login.get('secret_key'), self.auto_login.get('master_password')]:
            module.fail_json(msg='Unable to perform initial sign in to 1Password. subdomain, username, secret_key, and master_password are required to perform initial sign in.')
        args = ['signin', '{0}.1password.com'.format(self.auto_login['subdomain']), to_bytes(self.auto_login['username']), to_bytes(self.auto_login['secret_key']), '--output=raw']
        try:
            rc, out, err = self._run(args, command_input=to_bytes(self.auto_login['master_password']))
            self.token = out.strip()
        except AnsibleModuleError as e:
            module.fail_json(msg='Failed to perform initial sign in to 1Password: %s' % to_native(e))
    else:
        module.fail_json(msg="Unable to perform an initial sign in to 1Password. Please run '%s signin' or define credentials in 'auto_login'. See the module documentation for details." % self.cli_path)
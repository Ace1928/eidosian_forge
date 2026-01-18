from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def signin(self):
    self._check_required_params(['master_password'])
    args = ['signin', '--raw']
    if self.subdomain:
        args.extend(['--account', self.subdomain])
    return self._run(args, command_input=to_bytes(self.master_password))
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
def assert_logged_in(self):
    logged_in = self._cli.assert_logged_in()
    if logged_in:
        self.logged_in = logged_in
        pass
    else:
        self.set_token()
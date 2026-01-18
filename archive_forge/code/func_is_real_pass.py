from __future__ import (absolute_import, division, print_function)
from contextlib import contextmanager
import os
import re
import subprocess
import time
import yaml
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
from ansible.utils.encrypt import random_password
from ansible.plugins.lookup import LookupBase
from ansible import constants as C
from ansible_collections.community.general.plugins.module_utils._filelock import FileLock
def is_real_pass(self):
    if self.realpass is None:
        try:
            passoutput = to_text(check_output2([self.pass_cmd, '--version'], env=self.env), errors='surrogate_or_strict')
            self.realpass = 'pass: the standard unix password manager' in passoutput
        except subprocess.CalledProcessError as e:
            raise AnsibleError('exit code {0} while running {1}. Error output: {2}'.format(e.returncode, e.cmd, e.output))
    return self.realpass
from __future__ import absolute_import, division, print_function
import ctypes.util
import grp
import calendar
import os
import re
import pty
import pwd
import select
import shutil
import socket
import subprocess
import time
import math
from ansible.module_utils import distro
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
import ansible.module_utils.compat.typing as t
def check_password_encrypted(self):
    if self.module.params['password'] and self.platform != 'Darwin':
        maybe_invalid = False
        if self.module.params['password'] in set(['*', '!', '*************']):
            maybe_invalid = False
        else:
            if any((char in self.module.params['password'] for char in ':*!')):
                maybe_invalid = True
            if '$' not in self.module.params['password']:
                maybe_invalid = True
            else:
                fields = self.module.params['password'].split('$')
                if len(fields) >= 3:
                    if bool(_HASH_RE.search(fields[-1])):
                        maybe_invalid = True
                    if fields[1] == '1' and len(fields[-1]) != 22:
                        maybe_invalid = True
                    if fields[1] == '5' and len(fields[-1]) != 43:
                        maybe_invalid = True
                    if fields[1] == '6' and len(fields[-1]) != 86:
                        maybe_invalid = True
                else:
                    maybe_invalid = True
        if maybe_invalid:
            self.module.warn("The input password appears not to have been hashed. The 'password' argument must be encrypted for this module to work properly.")
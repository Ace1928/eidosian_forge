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
def ssh_key_fingerprint(self):
    ssh_key_file = self.get_ssh_key_path()
    if not os.path.exists(ssh_key_file):
        return (1, 'SSH Key file %s does not exist' % ssh_key_file, '')
    cmd = [self.module.get_bin_path('ssh-keygen', True)]
    cmd.append('-l')
    cmd.append('-f')
    cmd.append(ssh_key_file)
    return self.execute_command(cmd, obey_checkmode=False)
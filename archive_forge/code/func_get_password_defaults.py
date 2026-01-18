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
def get_password_defaults(self):
    try:
        minweeks = ''
        maxweeks = ''
        warnweeks = ''
        with open('/etc/default/passwd', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                m = re.match('^([^#]*)#(.*)$', line)
                if m:
                    line = m.group(1)
                key, value = line.split('=')
                if key == 'MINWEEKS':
                    minweeks = value.rstrip('\n')
                elif key == 'MAXWEEKS':
                    maxweeks = value.rstrip('\n')
                elif key == 'WARNWEEKS':
                    warnweeks = value.rstrip('\n')
    except Exception as err:
        self.module.fail_json(msg='failed to read /etc/default/passwd: %s' % to_native(err))
    return (minweeks, maxweeks, warnweeks)
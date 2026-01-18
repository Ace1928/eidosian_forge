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
def parse_shadow_file(self):
    """Example AIX shadowfile data:
        nobody:
                password = *

        operator1:
                password = {ssha512}06$xxxxxxxxxxxx....
                lastupdate = 1549558094

        test1:
                password = *
                lastupdate = 1553695126

        """
    b_name = to_bytes(self.name)
    b_passwd = b''
    b_expires = b''
    if os.path.exists(self.SHADOWFILE) and os.access(self.SHADOWFILE, os.R_OK):
        with open(self.SHADOWFILE, 'rb') as bf:
            b_lines = bf.readlines()
        b_passwd_line = b''
        b_expires_line = b''
        try:
            for index, b_line in enumerate(b_lines):
                if b_line.startswith(b'%s:' % b_name):
                    b_passwd_line = b_lines[index + 1]
                    b_expires_line = b_lines[index + 2]
                    break
            if b' = ' in b_passwd_line:
                b_passwd = b_passwd_line.split(b' = ', 1)[-1].strip()
            if b' = ' in b_expires_line:
                b_expires = b_expires_line.split(b' = ', 1)[-1].strip()
        except IndexError:
            self.module.fail_json(msg='Failed to parse shadow file %s' % self.SHADOWFILE)
    passwd = to_native(b_passwd)
    expires = to_native(b_expires) or -1
    return (passwd, expires)
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
def create_user_useradd(self, command_name='useradd'):
    cmd = [self.module.get_bin_path(command_name, True)]
    if self.uid is not None:
        cmd.append('-u')
        cmd.append(self.uid)
    if self.group is not None:
        if not self.group_exists(self.group):
            self.module.fail_json(msg='Group %s does not exist' % self.group)
        cmd.append('-g')
        cmd.append(self.group)
    if self.groups is not None and len(self.groups):
        groups = self.get_groups_set()
        cmd.append('-G')
        cmd.append(','.join(groups))
    if self.comment is not None:
        cmd.append('-c')
        cmd.append(self.comment)
    if self.home is not None:
        cmd.append('-d')
        cmd.append(self.home)
    if self.shell is not None:
        cmd.append('-s')
        cmd.append(self.shell)
    if self.create_home:
        cmd.append('-m')
        if self.skeleton is not None:
            cmd.append('-k')
            cmd.append(self.skeleton)
        if self.umask is not None:
            cmd.append('-K')
            cmd.append('UMASK=' + self.umask)
    cmd.append(self.name)
    rc, out, err = self.execute_command(cmd)
    if self.password is not None:
        cmd = []
        cmd.append(self.module.get_bin_path('chpasswd', True))
        cmd.append('-e')
        cmd.append('-c')
        self.execute_command(cmd, data='%s:%s' % (self.name, self.password))
    return (rc, out, err)
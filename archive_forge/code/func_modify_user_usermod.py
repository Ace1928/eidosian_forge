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
def modify_user_usermod(self):
    cmd = [self.module.get_bin_path('usermod', True)]
    info = self.user_info()
    if self.uid is not None and info[2] != int(self.uid):
        cmd.append('-u')
        cmd.append(self.uid)
    if self.group is not None:
        if not self.group_exists(self.group):
            self.module.fail_json(msg='Group %s does not exist' % self.group)
        ginfo = self.group_info(self.group)
        if info[3] != ginfo[2]:
            cmd.append('-g')
            cmd.append(self.group)
    if self.groups is not None:
        current_groups = self.user_group_membership()
        groups_need_mod = False
        groups = []
        if self.groups == '':
            if current_groups and (not self.append):
                groups_need_mod = True
        else:
            groups = self.get_groups_set(names_only=True)
            group_diff = set(current_groups).symmetric_difference(groups)
            if group_diff:
                if self.append:
                    for g in groups:
                        if g in group_diff:
                            groups_need_mod = True
                            break
                else:
                    groups_need_mod = True
        if groups_need_mod:
            cmd.append('-G')
            cmd.append(','.join(groups))
    if self.comment is not None and info[4] != self.comment:
        cmd.append('-c')
        cmd.append(self.comment)
    if self.home is not None and info[5] != self.home:
        if self.move_home:
            cmd.append('-m')
        cmd.append('-d')
        cmd.append(self.home)
    if self.shell is not None and info[6] != self.shell:
        cmd.append('-s')
        cmd.append(self.shell)
    if len(cmd) == 1:
        rc, out, err = (None, '', '')
    else:
        cmd.append(self.name)
        rc, out, err = self.execute_command(cmd)
    if self.update_password == 'always' and self.password is not None and (info[1] != self.password):
        cmd = []
        cmd.append(self.module.get_bin_path('chpasswd', True))
        cmd.append('-e')
        cmd.append('-c')
        rc2, out2, err2 = self.execute_command(cmd, data='%s:%s' % (self.name, self.password))
    else:
        rc2, out2, err2 = (None, '', '')
    if rc is not None:
        return (rc, out + out2, err + err2)
    else:
        return (rc2, out + out2, err + err2)
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
def modify_user(self):
    current_groups = self.user_group_membership()
    groups = []
    rc = None
    out = ''
    err = ''
    info = self.user_info()
    add_cmd_bin = self.module.get_bin_path('adduser', True)
    remove_cmd_bin = self.module.get_bin_path('delgroup', True)
    if self.groups is not None and len(self.groups):
        groups = self.get_groups_set()
        group_diff = set(current_groups).symmetric_difference(groups)
        if group_diff:
            for g in groups:
                if g in group_diff:
                    add_cmd = [add_cmd_bin, self.name, g]
                    rc, out, err = self.execute_command(add_cmd)
                    if rc is not None and rc != 0:
                        self.module.fail_json(name=self.name, msg=err, rc=rc)
            for g in group_diff:
                if g not in groups and (not self.append):
                    remove_cmd = [remove_cmd_bin, self.name, g]
                    rc, out, err = self.execute_command(remove_cmd)
                    if rc is not None and rc != 0:
                        self.module.fail_json(name=self.name, msg=err, rc=rc)
    if self.update_password == 'always' and self.password is not None and (info[1] != self.password):
        cmd = [self.module.get_bin_path('chpasswd', True)]
        cmd.append('--encrypted')
        data = '{name}:{password}'.format(name=self.name, password=self.password)
        rc, out, err = self.execute_command(cmd, data=data)
        if rc is not None and rc != 0:
            self.module.fail_json(name=self.name, msg=err, rc=rc)
    return (rc, out, err)
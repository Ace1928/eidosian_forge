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
def get_groups_set(self, remove_existing=True, names_only=False):
    if self.groups is None:
        return None
    info = self.user_info()
    groups = set((x.strip() for x in self.groups.split(',') if x))
    group_names = set()
    for g in groups.copy():
        if not self.group_exists(g):
            self.module.fail_json(msg='Group %s does not exist' % g)
        group_info = self.group_info(g)
        if info and remove_existing and (group_info[2] == info[3]):
            groups.remove(g)
        elif names_only:
            group_names.add(group_info[0])
    if names_only:
        return group_names
    return groups
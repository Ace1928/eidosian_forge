from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
def get_installed(self):
    cmd = ['install', '--list']
    if self.path:
        cmd.append('--root')
        cmd.append(self.path)
    data, dummy = self._exec(cmd, True, False, False)
    package_regex = re.compile('^([\\w\\-]+) v(.+):$')
    installed = {}
    for line in data.splitlines():
        package_info = package_regex.match(line)
        if package_info:
            installed[package_info.group(1)] = package_info.group(2)
    return installed
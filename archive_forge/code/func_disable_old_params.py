from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def disable_old_params(self):
    for modprobe_file in self.modprobe_files:
        with open(modprobe_file) as file:
            file_content = file.readlines()
        content_changed = False
        for index, line in enumerate(file_content):
            if self.re_find_params.match(line):
                file_content[index] = '#' + line
                content_changed = True
        if content_changed:
            with open(modprobe_file, 'w') as file:
                file.write('\n'.join(file_content))
from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def permanent_params(self):
    params = set()
    for modprobe_file in self.modprobe_files:
        with open(modprobe_file) as file:
            for line in file:
                match = self.re_get_params_and_values.match(line)
                if match:
                    params.add(match.group(1))
    return params
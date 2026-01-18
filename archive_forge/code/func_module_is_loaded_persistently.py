from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def module_is_loaded_persistently(self):
    for module_file in self.modules_files:
        with open(module_file) as file:
            for line in file:
                if self.re_find_module.match(line):
                    return True
    return False
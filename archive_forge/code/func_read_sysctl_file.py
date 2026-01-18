from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def read_sysctl_file(self):
    lines = []
    if os.path.isfile(self.sysctl_file):
        try:
            with open(self.sysctl_file, 'r') as read_file:
                lines = read_file.readlines()
        except IOError as e:
            self.module.fail_json(msg='Failed to open %s: %s' % (to_native(self.sysctl_file), to_native(e)))
    for line in lines:
        line = line.strip()
        self.file_lines.append(line)
        if not line or line.startswith(('#', ';')) or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip()
        self.file_values[k] = v.strip()
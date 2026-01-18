from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def fix_lines(self):
    checked = []
    self.fixed_lines = []
    for line in self.file_lines:
        if not line.strip() or line.strip().startswith(('#', ';')) or '=' not in line:
            self.fixed_lines.append(line)
            continue
        tmpline = line.strip()
        k, v = tmpline.split('=', 1)
        k = k.strip()
        v = v.strip()
        if k not in checked:
            checked.append(k)
            if k == self.args['name']:
                if self.args['state'] == 'present':
                    new_line = '%s=%s\n' % (k, self.args['value'])
                    self.fixed_lines.append(new_line)
            else:
                new_line = '%s=%s\n' % (k, v)
                self.fixed_lines.append(new_line)
    if self.args['name'] not in checked and self.args['state'] == 'present':
        new_line = '%s=%s\n' % (self.args['name'], self.args['value'])
        self.fixed_lines.append(new_line)
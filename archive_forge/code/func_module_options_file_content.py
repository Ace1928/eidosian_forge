from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def module_options_file_content(self):
    file_content = ['options {0} {1}'.format(self.name, param) for param in self.params.split()]
    return '\n'.join(file_content) + '\n'
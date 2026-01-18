from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def create_module_options_file(self):
    new_file_path = os.path.join(PARAMETERS_FILES_LOCATION, self.name + '.conf')
    with open(new_file_path, 'w') as file:
        file.write(self.module_options_file_content)
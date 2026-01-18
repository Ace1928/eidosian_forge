from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def load_module_permanent(self):
    if not self.module_is_loaded_persistently:
        self.create_module_file()
        self.changed = True
    if not self.params_is_set:
        self.disable_old_params()
        self.create_module_options_file()
        self.changed = True
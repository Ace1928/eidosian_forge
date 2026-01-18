from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def module_loaded(self):
    is_loaded = False
    try:
        with open('/proc/modules') as modules:
            module_name = self.name.replace('-', '_') + ' '
            for line in modules:
                if line.startswith(module_name):
                    is_loaded = True
                    break
        if not is_loaded:
            module_file = '/' + self.name + '.ko'
            builtin_path = os.path.join('/lib/modules/', RELEASE_VER, 'modules.builtin')
            with open(builtin_path) as builtins:
                for line in builtins:
                    if line.rstrip().endswith(module_file):
                        is_loaded = True
                        break
    except (IOError, OSError) as e:
        self.module.fail_json(msg=to_native(e), exception=traceback.format_exc(), **self.result)
    return is_loaded
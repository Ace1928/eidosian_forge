from __future__ import (absolute_import, division, print_function)
import os
import os.path
import random
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import text_type, string_types
from ansible.plugins import AnsiblePlugin
def build_module_command(self, env_string, shebang, cmd, arg_path=None):
    if cmd.strip() != '':
        cmd = shlex.quote(cmd)
    cmd_parts = []
    if shebang:
        shebang = shebang.replace('#!', '').strip()
    else:
        shebang = ''
    cmd_parts.extend([env_string.strip(), shebang, cmd])
    if arg_path is not None:
        cmd_parts.append(arg_path)
    new_cmd = ' '.join(cmd_parts)
    return new_cmd
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
def set_user_facl(self, paths, user, mode):
    """Only sets acls for users as that's really all we need"""
    cmd = ['setfacl', '-m', 'u:%s:%s' % (user, mode)]
    cmd.extend(paths)
    cmd = [shlex.quote(c) for c in cmd]
    return ' '.join(cmd)
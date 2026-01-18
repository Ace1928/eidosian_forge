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
def append_command(self, cmd, cmd_to_append):
    """Append an additional command if supported by the shell"""
    if self._SHELL_AND:
        cmd += ' %s %s' % (self._SHELL_AND, cmd_to_append)
    return cmd
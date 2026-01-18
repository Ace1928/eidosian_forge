from __future__ import (absolute_import, division, print_function)
import os
import os.path
import subprocess
import traceback
from ansible.errors import AnsibleError
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
def list_jails(self):
    p = subprocess.Popen([self.jls_cmd, '-q', 'name'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return to_text(stdout, errors='surrogate_or_strict').split()
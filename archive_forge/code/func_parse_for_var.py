from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import shlex
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
def parse_for_var(self, line):
    lexer = shlex.shlex(line)
    lexer.wordchars = self.wordchars
    varname = lexer.get_token()
    is_env_var = lexer.get_token() == '='
    value = ''.join(lexer)
    if is_env_var:
        return (varname, value)
    raise CronVarError('Not a variable.')
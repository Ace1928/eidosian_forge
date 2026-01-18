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
def find_variable(self, name):
    for l in self.lines:
        try:
            varname, value = self.parse_for_var(l)
            if varname == name:
                return value
        except CronVarError:
            pass
    return None
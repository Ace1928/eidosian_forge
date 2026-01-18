from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def get_token_curr_value(self, token):
    if self.platform == 'openbsd':
        thiscmd = '%s -n %s' % (self.sysctl_cmd, token)
    else:
        thiscmd = '%s -e -n %s' % (self.sysctl_cmd, token)
    rc, out, err = self.module.run_command(thiscmd, environ_update=self.LANG_ENV)
    if rc != 0:
        return None
    else:
        return out
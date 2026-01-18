from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def set_token_value(self, token, value):
    if len(value.split()) > 0:
        value = '"' + value + '"'
    if self.platform == 'openbsd':
        thiscmd = '%s %s=%s' % (self.sysctl_cmd, token, value)
    elif self.platform == 'freebsd':
        ignore_missing = ''
        if self.args['ignoreerrors']:
            ignore_missing = '-i'
        thiscmd = '%s %s %s=%s' % (self.sysctl_cmd, ignore_missing, token, value)
    else:
        ignore_missing = ''
        if self.args['ignoreerrors']:
            ignore_missing = '-e'
        thiscmd = '%s %s -w %s=%s' % (self.sysctl_cmd, ignore_missing, token, value)
    rc, out, err = self.module.run_command(thiscmd, environ_update=self.LANG_ENV)
    if rc != 0 or self._stderr_failed(err):
        self.module.fail_json(msg='setting %s failed: %s' % (token, out + err))
    else:
        return rc
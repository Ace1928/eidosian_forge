from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils._text import to_native
def reload_sysctl(self):
    if self.platform == 'freebsd':
        rc, out, err = self.module.run_command('/etc/rc.d/sysctl reload', environ_update=self.LANG_ENV)
    elif self.platform == 'openbsd':
        for k, v in self.file_values.items():
            rc = 0
            if k != self.args['name']:
                rc = self.set_token_value(k, v)
                if rc != 0:
                    break
        if rc == 0 and self.args['state'] == 'present':
            rc = self.set_token_value(self.args['name'], self.args['value'])
        return
    else:
        sysctl_args = [self.sysctl_cmd, '-p', self.sysctl_file]
        if self.args['ignoreerrors']:
            sysctl_args.insert(1, '-e')
        rc, out, err = self.module.run_command(sysctl_args, environ_update=self.LANG_ENV)
    if rc != 0 or self._stderr_failed(err):
        self.module.fail_json(msg='Failed to reload sysctl: %s' % to_native(out) + to_native(err))
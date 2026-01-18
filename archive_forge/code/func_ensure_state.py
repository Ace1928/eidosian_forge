from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from fnmatch import fnmatch
def ensure_state(self, packages, command):
    """ Ensure packages state """
    rc, out, err = self.module.run_command([self.yum_bin, '-q', 'versionlock', command] + packages)
    if 'No package found for' in out:
        self.module.fail_json(msg=out)
    if rc == 0:
        return True
    self.module.fail_json(msg='Error: ' + to_native(err) + to_native(out))
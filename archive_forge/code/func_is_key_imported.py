from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def is_key_imported(self, keyid):
    cmd = self.rpm + ' -q  gpg-pubkey'
    rc, stdout, stderr = self.module.run_command(cmd)
    if rc != 0:
        return False
    cmd += ' --qf "%{description}" | ' + self.gpg + ' --no-tty --batch --with-colons --fixed-list-mode -'
    stdout, stderr = self.execute_command(cmd)
    for line in stdout.splitlines():
        if keyid in line.split(':')[4]:
            return True
    return False
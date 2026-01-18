from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def lsign_key(self, keyring, keyid):
    """Locally sign key"""
    cmd = [self.pacman_key, '--gpgdir', keyring]
    self.module.run_command(cmd + ['--lsign-key', keyid], check_rc=True)
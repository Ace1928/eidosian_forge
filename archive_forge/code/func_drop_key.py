from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def drop_key(self, keyid):
    if not self.module.check_mode:
        self.execute_command([self.rpm, '--erase', '--allmatches', 'gpg-pubkey-%s' % keyid[-8:].lower()])
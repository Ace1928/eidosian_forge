from __future__ import (absolute_import, division, print_function)
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def save_key(self, data):
    """Saves key data to a temporary file"""
    tmpfd, tmpname = tempfile.mkstemp()
    self.module.add_cleanup_file(tmpname)
    tmpfile = os.fdopen(tmpfd, 'w')
    tmpfile.write(data)
    tmpfile.close()
    return tmpname
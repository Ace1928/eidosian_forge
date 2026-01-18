from __future__ import absolute_import, division, print_function
import os
import platform
import re
import stat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_mountpoint(self):
    """Return (first) mountpoint of device. Returns None when not mounted."""
    cmd_findmnt = self.module.get_bin_path('findmnt', required=True)
    rc, mountpoint, dummy = self.module.run_command([cmd_findmnt, '--mtab', '--noheadings', '--output', 'TARGET', '--source', self.path], check_rc=False)
    if rc != 0:
        mountpoint = None
    else:
        mountpoint = mountpoint.split('\n')[0]
    return mountpoint
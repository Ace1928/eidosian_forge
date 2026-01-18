from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def run_luks_remove(self, device):
    wipefs_bin = self._module.get_bin_path('wipefs', True)
    name = self.get_container_name_by_device(device)
    if name is not None:
        self.run_luks_close(name)
    result = self._run_command([wipefs_bin, '--all', device])
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while wiping LUKS container signatures for %s: %s' % (device, result[STDERR]))
    try:
        wipe_luks_headers(device)
    except Exception as exc:
        raise ValueError('Error while wiping LUKS container signatures for %s: %s' % (device, exc))
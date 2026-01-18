from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def get_container_name_by_device(self, device):
    """ obtain LUKS container name based on the device where it is located
            return None if not found
            raise ValueError if lsblk command fails
        """
    result = self._run_command([self._lsblk_bin, device, '-nlo', 'type,name'])
    if result[RETURN_CODE] != 0:
        raise ValueError('Error while obtaining LUKS name for %s: %s' % (device, result[STDERR]))
    for line in result[STDOUT].splitlines(False):
        m = LUKS_NAME_REGEX.match(line)
        if m:
            return m.group(1)
    return None
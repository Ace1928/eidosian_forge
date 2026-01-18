from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def is_luks(self, device):
    """ check if the LUKS container does exist
        """
    result = self._run_command([self._cryptsetup_bin, 'isLuks', device])
    return result[RETURN_CODE] == 0
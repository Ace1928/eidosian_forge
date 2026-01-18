from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_any_mountpoint(self):
    for subvol_mountpoints in self.__mountpoints.values():
        if len(subvol_mountpoints) > 0:
            return subvol_mountpoints[0]
    return None
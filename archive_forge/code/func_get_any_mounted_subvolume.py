from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_any_mounted_subvolume(self):
    for subvolid, subvol_mountpoints in self.__mountpoints.items():
        if len(subvol_mountpoints) > 0:
            return self.get_subvolume_by_id(subvolid)
    return None
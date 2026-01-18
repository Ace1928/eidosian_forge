from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_mounted_path(self):
    mountpoints = self.get_mountpoints()
    if mountpoints is not None and len(mountpoints) > 0:
        return mountpoints[0]
    elif self.parent is not None:
        parent = self.__filesystem.get_subvolume_by_id(self.parent)
        parent_path = parent.get_mounted_path()
        if parent_path is not None:
            return parent_path + os.path.sep + self.name
    else:
        return None
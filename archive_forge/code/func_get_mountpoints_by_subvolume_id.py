from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_mountpoints_by_subvolume_id(self, subvolume_id):
    return self.__mountpoints[subvolume_id] if subvolume_id in self.__mountpoints else []
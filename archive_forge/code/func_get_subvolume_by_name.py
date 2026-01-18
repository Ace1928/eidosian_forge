from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_subvolume_by_name(self, subvolume):
    for subvolume_info in self.__subvolumes.values():
        if subvolume_info['path'] == subvolume:
            return BtrfsSubvolume(self, subvolume_info['id'])
    return None
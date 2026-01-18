from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_nearest_subvolume(self, subvolume):
    """Return the identified subvolume if existing, else the closest matching parent"""
    subvolumes_by_path = self.__get_subvolumes_by_path()
    while len(subvolume) > 1:
        if subvolume in subvolumes_by_path:
            return BtrfsSubvolume(self, subvolumes_by_path[subvolume]['id'])
        else:
            subvolume = re.sub('/[^/]+$', '', subvolume)
    return BtrfsSubvolume(self, 5)
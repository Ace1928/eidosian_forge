from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_child_subvolumes(self):
    return self.__filesystem.get_subvolume_children(self.__subvolume_id)
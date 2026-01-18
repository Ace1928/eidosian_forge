from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def is_filesystem_default(self):
    return self.__filesystem.default_subvolid == self.__subvolume_id
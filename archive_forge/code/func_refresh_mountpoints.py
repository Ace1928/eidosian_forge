from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def refresh_mountpoints(self):
    mountpoints = self.__provider.get_mountpoints(list(self.__devices))
    self.__update_mountpoints(mountpoints)
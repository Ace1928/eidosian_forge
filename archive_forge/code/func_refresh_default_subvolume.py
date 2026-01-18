from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def refresh_default_subvolume(self):
    filesystem_path = self.get_any_mountpoint()
    if filesystem_path is not None:
        self.__default_subvolid = self.__provider.get_default_subvolume_id(filesystem_path)
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def subvolume_delete(self, subvolume_path):
    command = [self.__btrfs, 'subvolume', 'delete', to_bytes(subvolume_path)]
    result = self.__module.run_command(command, check_rc=True)
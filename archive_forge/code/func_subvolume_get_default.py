from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def subvolume_get_default(self, filesystem_path):
    command = [self.__btrfs, 'subvolume', 'get-default', to_bytes(filesystem_path)]
    result = self.__module.run_command(command, check_rc=True)
    return int(result[1].strip().split()[1])
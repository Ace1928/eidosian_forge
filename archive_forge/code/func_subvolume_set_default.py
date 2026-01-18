from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def subvolume_set_default(self, filesystem_path, subvolume_id):
    command = [self.__btrfs, 'subvolume', 'set-default', str(subvolume_id), to_bytes(filesystem_path)]
    result = self.__module.run_command(command, check_rc=True)
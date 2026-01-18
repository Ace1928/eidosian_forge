from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def subvolumes_list(self, filesystem_path):
    command = '%s subvolume list -tap %s' % (self.__btrfs, filesystem_path)
    result = self.__module.run_command(command, check_rc=True)
    stdout = [x.split('\t') for x in result[1].splitlines()]
    subvolumes = [{'id': 5, 'parent': None, 'path': '/'}]
    if len(stdout) > 2:
        subvolumes.extend([self.__parse_subvolume_list_record(x) for x in stdout[2:]])
    return subvolumes
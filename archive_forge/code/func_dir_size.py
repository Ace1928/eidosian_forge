from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import (
from ansible.module_utils.common.text.converters import to_native
def dir_size(module, path):
    total_size = 0
    for path, dirs, files in os.walk(path):
        for f in files:
            total_size += os.path.getsize(os.path.join(path, f))
    return total_size
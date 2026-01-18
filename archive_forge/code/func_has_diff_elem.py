from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def has_diff_elem(ls1, ls2):
    return any((elem not in ls1 for elem in ls2))
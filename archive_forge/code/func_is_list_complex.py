from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def is_list_complex(x):
    return isinstance(x[0], dict) or isinstance(x[0], list)
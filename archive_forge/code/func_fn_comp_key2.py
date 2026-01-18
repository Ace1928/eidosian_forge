from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def fn_comp_key2(k, dict1, dict2):
    return dnac_compare_equality2(dict1.get(k), dict2.get(k))
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def get_dict_result(result, key, value, cmp_fn=simple_cmp):
    if isinstance(result, list):
        if len(result) == 1:
            if isinstance(result[0], dict):
                result = result[0]
                if result.get(key) is not None and result.get(key) != value:
                    result = None
            else:
                result = None
        else:
            for item in result:
                if isinstance(item, dict) and (item.get(key) is None or item.get(key) == value):
                    result = item
                    return result
            result = None
    elif not isinstance(result, dict):
        result = None
    elif result.get(key) is not None and result.get(key) != value:
        result = None
    return result
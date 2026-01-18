from __future__ import (absolute_import, division, print_function)
import re
def is_same_comparison(reorder_current, reorder_filtered):
    for key, value in reorder_filtered.items():
        if key not in reorder_current:
            return False
        if isinstance(value, dict):
            if not is_same_comparison(reorder_current[key], value):
                return False
        elif isinstance(value, list):
            if len(value) != len(reorder_current[key]):
                return False
            if len(value) and isinstance(value[0], dict):
                for current_dict in reorder_current[key]:
                    if not is_same_comparison(current_dict, value[0]):
                        return False
            elif reorder_current[key] != value:
                return False
        elif isinstance(value, str) and IP_PREFIX.match(value):
            return is_same_ip_address(reorder_current[key], value)
        elif reorder_current[key] != value:
            return False
    return True
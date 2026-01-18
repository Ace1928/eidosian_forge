from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import string_types
def pct_to_int(value, num_items, min_value=1):
    """
    Converts a given value to a percentage if specified as "x%",
    otherwise converts the given value to an integer.
    """
    if isinstance(value, string_types) and value.endswith('%'):
        value_pct = int(value.replace('%', ''))
        return int(value_pct / 100.0 * num_items) or min_value
    else:
        return int(value)
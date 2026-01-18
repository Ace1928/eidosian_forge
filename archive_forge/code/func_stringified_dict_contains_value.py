import re
from kombu.utils.encoding import safe_str
def stringified_dict_contains_value(key, value, str_dict):
    """Checks if dict in for of string like "{'test': 5}" contains
    key/value pair. This works faster, then creating actual dict
    from string since this operation is called for each task in case
    of kwargs search."""
    if not str_dict:
        return False
    value = str(value)
    try:
        key_index = str_dict.index(key) + len(key) + 3
    except ValueError:
        return False
    try:
        comma_index = str_dict.index(',', key_index)
    except ValueError:
        comma_index = str_dict.index('}', key_index)
    return str(value) == str_dict[key_index:comma_index].strip('"\'')
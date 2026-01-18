from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def single_item_dicts_to_dict(data):
    result = {}
    for item in data:
        (k, v), = item.items()
        result[k] = v
    return result
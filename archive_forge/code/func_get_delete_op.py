from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_delete_op(key, test_keys):
    del_op = __DELETE_OP_DEFAULT
    if not test_keys:
        return del_op
    if not key:
        key = '__default_ops'
    t_keys = next((t_key_item[key] for t_key_item in test_keys if key in t_key_item), None)
    if t_keys:
        del_op = t_keys.get('__delete_op', __DELETE_OP_DEFAULT)
    return del_op
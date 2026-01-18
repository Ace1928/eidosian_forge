from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def in_place_merge(a, b):
    """
    Recursively merges second dict into the first.

    """
    if not isinstance(b, dict):
        return b
    for k, v in b.items():
        if k in a and isinstance(a[k], dict):
            a[k] = in_place_merge(a[k], v)
        else:
            a[k] = v
    return a
from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def is_object_ref(d):
    """
    Checks if a dictionary is a reference object. The dictionary is considered to be a
    reference object when it contains non-empty 'id' and 'type' fields.

    :type d: dict
    :return: True if passed dictionary is a reference object, otherwise False
    """
    has_id = 'id' in d.keys() and d['id']
    has_type = 'type' in d.keys() and d['type']
    return has_id and has_type
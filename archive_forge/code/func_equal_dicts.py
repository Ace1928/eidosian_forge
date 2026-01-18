from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def equal_dicts(d1, d2, compare_by_reference=True):
    """
    Checks whether two dictionaries are equal. If `compare_by_reference` is set to True, dictionaries referencing
    objects are compared using `equal_object_refs` method. Otherwise, every key and value is checked.

    :type d1: dict
    :type d2: dict
    :param compare_by_reference: if True, dictionaries referencing objects are compared using `equal_object_refs` method
    :return: True if passed dicts are equal. Otherwise, returns False.
    """
    if compare_by_reference and is_object_ref(d1) and is_object_ref(d2):
        return equal_object_refs(d1, d2)
    if len(d1) != len(d2):
        return False
    for key, v1 in d1.items():
        if key not in d2:
            return False
        v2 = d2[key]
        if not equal_values(v1, v2):
            return False
    return True
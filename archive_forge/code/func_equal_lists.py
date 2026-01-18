from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def equal_lists(l1, l2):
    """
    Checks whether two lists are equal. The order of elements in the arrays is important.

    :type l1: list
    :type l2: list
    :return: True if passed lists, their elements and order of elements are equal. Otherwise, returns False.
    """
    if len(l1) != len(l2):
        return False
    for v1, v2 in zip(l1, l2):
        if not equal_values(v1, v2):
            return False
    return True
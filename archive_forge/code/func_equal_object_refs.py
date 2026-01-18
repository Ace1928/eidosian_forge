from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def equal_object_refs(d1, d2):
    """
    Checks whether two references point to the same object.

    :type d1: dict
    :type d2: dict
    :return: True if passed references point to the same object, otherwise False
    """
    have_equal_ids = d1['id'] == d2['id']
    have_equal_types = d1['type'] == d2['type']
    return have_equal_ids and have_equal_types
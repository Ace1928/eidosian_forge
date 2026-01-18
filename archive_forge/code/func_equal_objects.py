from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def equal_objects(d1, d2):
    """
    Checks whether two objects are equal. Ignores special object properties (e.g. 'id', 'version') and
    properties with None and empty values. In case properties contains a reference to the other object,
    only object identities (ids and types) are checked. Also, if an array field contains multiple references
    to the same object, duplicates are ignored when comparing objects.

    :type d1: dict
    :type d2: dict
    :return: True if passed objects and their properties are equal. Otherwise, returns False.
    """

    def prepare_data_for_comparison(d):
        d = dict(((k, d[k]) for k in d.keys() if k not in NON_COMPARABLE_PROPERTIES and d[k]))
        d = delete_ref_duplicates(d)
        return d
    d1 = prepare_data_for_comparison(d1)
    d2 = prepare_data_for_comparison(d2)
    return equal_dicts(d1, d2, compare_by_reference=False)
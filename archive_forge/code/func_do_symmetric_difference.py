from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def do_symmetric_difference(a, b):
    sym_diff = []
    union = lists_union(a, b)
    try:
        isect = set(a) & set(b)
        sym_diff = [item for item in union if item not in isect]
    except TypeError:
        isect = lists_intersect(a, b)
        sym_diff = [item for item in union if item not in isect]
    return sym_diff
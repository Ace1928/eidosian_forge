from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def do_difference(a, b):
    diff = []
    try:
        other = set(b)
        diff = [item for item in a if item not in other]
    except TypeError:
        other = list(b)
        diff = [item for item in a if item not in other]
    return diff
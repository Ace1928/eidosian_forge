from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
def cmp_simple_list(want, have):
    if want is None:
        return None
    if have is None and want in ['', 'none']:
        return None
    if have is not None and want in ['', 'none']:
        return []
    if have is None:
        return want
    if set(want) != set(have):
        return want
    return None
from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
def cmp_str_with_none(want, have):
    if want is None:
        return None
    if have is None and want == '':
        return None
    if want != have:
        return want
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def diff_list_of_dicts(want, have):
    diff = []
    set_w = set((tuple(d.items()) for d in want))
    set_h = set((tuple(d.items()) for d in have))
    difference = set_w.difference(set_h)
    for element in difference:
        diff.append(dict(((x, y) for x, y in element)))
    return diff
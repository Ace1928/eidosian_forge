from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def lists_symmetric_difference(*args, **kwargs):
    lists = args
    flatten = kwargs.pop('flatten', False)
    if kwargs:
        raise AnsibleFilterError('lists_difference() got unexpected keywords arguments: {0}'.format(', '.join(kwargs.keys())))
    if flatten:
        lists = flatten_list(args)
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]
    a = lists[0]
    for b in lists[1:]:
        a = do_symmetric_difference(a, b)
    return a
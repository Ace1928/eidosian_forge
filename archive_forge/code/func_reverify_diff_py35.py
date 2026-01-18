from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def reverify_diff_py35(want, have):
    """Function to re-verify the set diff for py35 as it doesn't maintains dict order which results
        into unexpected set diff
    :param config: want and have set config
    :returns: True/False post checking if there's any actual diff b/w want and have sets
    """
    if not have:
        return True
    for each_want in want:
        diff = True
        for each_have in have:
            if each_have == sorted(each_want) or sorted(each_have) == sorted(each_want):
                diff = False
        if diff:
            return True
    return False
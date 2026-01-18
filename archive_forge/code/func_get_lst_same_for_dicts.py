from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def get_lst_same_for_dicts(want, have, lst):
    """
    This function generates a list containing values
    that are common for list in want and list in have dict
    :param want: dict object to want
    :param have: dict object to have
    :param lst: list the comparison on
    :return: new list object with values which are common in want and have.
    """
    diff = None
    if want and have:
        want_list = want.get(lst) or {}
        have_list = have.get(lst) or {}
        diff = [i for i in want_list and have_list if i in have_list and i in want_list]
    return diff
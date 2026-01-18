from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def search_dict_tv_in_list(d_val1, d_val2, lst, key1, key2):
    """
    This function return the dict object if it exist in list.
    :param d_val1:
    :param d_val2:
    :param lst:
    :param key1:
    :param key2:
    :return:
    """
    obj = next((item for item in lst if item[key1] == d_val1 and item[key2] == d_val2), None)
    if obj:
        return obj
    else:
        return None
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def key_value_in_dict(have_key, have_value, want_dict):
    """
    This function checks whether the key and values exist in dict
    :param have_key:
    :param have_value:
    :param want_dict:
    :return:
    """
    for key, value in iteritems(want_dict):
        if key == have_key and value == have_value:
            return True
    return False
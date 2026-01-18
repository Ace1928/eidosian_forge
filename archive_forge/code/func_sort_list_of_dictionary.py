from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def sort_list_of_dictionary(list_of_dict):
    """
    This functions sorts a list of dictionaries. It first sorts each value of the dictionary and then sorts the list of
    individually sorted dictionaries. For sorting, each dictionary's tuple equivalent is used.
    :param list_of_dict: List of dictionaries.
    :return: A sorted dictionary.
    """
    list_with_sorted_dict = []
    for d in list_of_dict:
        sorted_d = sort_dictionary(d)
        list_with_sorted_dict.append(sorted_d)
    return sorted(list_with_sorted_dict, key=get_key_for_comparing_dict)
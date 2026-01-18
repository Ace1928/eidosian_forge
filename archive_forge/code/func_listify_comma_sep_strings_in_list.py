from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
def listify_comma_sep_strings_in_list(self, some_list):
    """
        method to accept a list of strings as the parameter, find any strings
        in that list that are comma separated, remove them from the list and add
        their comma separated elements to the original list
        """
    new_list = []
    remove_from_original_list = []
    for element in some_list:
        if ',' in element:
            remove_from_original_list.append(element)
            new_list.extend([e.strip() for e in element.split(',')])
    for element in remove_from_original_list:
        some_list.remove(element)
    some_list.extend(new_list)
    if some_list == ['']:
        return []
    return some_list
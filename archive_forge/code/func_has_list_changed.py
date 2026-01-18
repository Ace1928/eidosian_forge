from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def has_list_changed(new_list, old_list, sort_lists=True, sort_key=None):
    """
    Check two lists have differences. Sort lists by default.
    """

    def sort_list(unsorted_list):
        """
        Sort a given list.
        The list may contain dictionaries, so use the sort key to handle them.
        """
        if unsorted_list and isinstance(unsorted_list[0], dict):
            if not sort_key:
                raise Exception('A sort key was not specified when sorting list')
            else:
                return sorted(unsorted_list, key=lambda k: k[sort_key])
        try:
            return sorted(unsorted_list)
        except TypeError:
            return unsorted_list
    if new_list is None:
        return False
    old_list = old_list or []
    if len(new_list) != len(old_list):
        return True
    if sort_lists:
        zip_data = zip(sort_list(new_list), sort_list(old_list))
    else:
        zip_data = zip(new_list, old_list)
    for new_item, old_item in zip_data:
        is_same_type = type(new_item) == type(old_item)
        if not is_same_type:
            if isinstance(new_item, string_types) and isinstance(old_item, string_types):
                try:
                    new_item_type = type(new_item)
                    old_item_casted = new_item_type(old_item)
                    if new_item != old_item_casted:
                        return True
                    else:
                        continue
                except UnicodeEncodeError:
                    return True
            else:
                return True
        if isinstance(new_item, dict):
            if has_dict_changed(new_item, old_item):
                return True
        elif new_item != old_item:
            return True
    return False
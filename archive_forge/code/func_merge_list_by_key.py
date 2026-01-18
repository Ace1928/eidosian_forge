from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
def merge_list_by_key(original_list, updated_list, key, ignore_when_null=None):
    """
    Merge two lists by the key. It basically:

    1. Adds the items that are present on updated_list and are absent on original_list.

    2. Removes items that are absent on updated_list and are present on original_list.

    3. For all items that are in both lists, overwrites the values from the original item by the updated item.

    :arg list original_list: original list.
    :arg list updated_list: list with changes.
    :arg str key: unique identifier.
    :arg list ignore_when_null: list with the keys from the updated items that should be ignored in the merge,
        if its values are null.
    :return: list: Lists merged.
    """
    ignore_when_null = [] if ignore_when_null is None else ignore_when_null
    if not original_list:
        return updated_list
    items_map = collections.OrderedDict([(i[key], i.copy()) for i in original_list])
    merged_items = collections.OrderedDict()
    for item in updated_list:
        item_key = item[key]
        if item_key in items_map:
            for ignored_key in ignore_when_null:
                if ignored_key in item and item[ignored_key] is None:
                    item.pop(ignored_key)
            merged_items[item_key] = items_map[item_key]
            merged_items[item_key].update(item)
        else:
            merged_items[item_key] = item
    return list(merged_items.values())
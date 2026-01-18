from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import json
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.kubernetes.core.plugins.module_utils.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def recursive_list_diff(list1, list2, position=None):
    result = (list(), list())
    if position in STRATEGIC_MERGE_PATCH_KEYS:
        patch_merge_key = STRATEGIC_MERGE_PATCH_KEYS[position]
        dict1 = list_to_dict(list1, patch_merge_key, position)
        dict2 = list_to_dict(list2, patch_merge_key, position)
        dict1_keys = set(dict1.keys())
        dict2_keys = set(dict2.keys())
        for key in dict1_keys - dict2_keys:
            result[0].append(dict1[key])
        for key in dict2_keys - dict1_keys:
            result[1].append(dict2[key])
        for key in dict1_keys & dict2_keys:
            diff = recursive_diff(dict1[key], dict2[key], position)
            if diff:
                diff[0].update({patch_merge_key: dict1[key][patch_merge_key]})
                diff[1].update({patch_merge_key: dict2[key][patch_merge_key]})
                result[0].append(diff[0])
                result[1].append(diff[1])
        if result[0] or result[1]:
            return result
    elif list1 != list2:
        return (list1, list2)
    return None
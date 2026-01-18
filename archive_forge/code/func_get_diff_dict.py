from __future__ import absolute_import, division, print_function
import re
import json
import ast
from copy import copy
from itertools import (count, groupby)
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.common.network import (
from ansible.module_utils.common.validation import check_required_arguments
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def get_diff_dict(base_data, compare_with_data, test_keys=None, is_skeleton=None):
    if is_skeleton is None:
        is_skeleton = False
    if test_keys is None:
        test_keys = []
    if not base_data:
        return base_data
    planned_set = set(base_data.keys())
    discovered_set = set(compare_with_data.keys())
    intersect_set = planned_set.intersection(discovered_set)
    changed_dict = {}
    has_dict_item = None
    added_set = planned_set - intersect_set
    if added_set:
        for key in added_set:
            if is_skeleton:
                changed_dict[key] = base_data[key]
            elif base_data[key] is not None:
                if isinstance(base_data[key], dict):
                    val_dict = remove_empties(base_data[key])
                    if val_dict:
                        changed_dict[key] = remove_empties(base_data[key])
                elif isinstance(base_data[key], list):
                    val_list = remove_empties_from_list(base_data[key])
                    if val_list:
                        changed_dict[key] = remove_empties_from_list(base_data[key])
                else:
                    changed_dict[key] = base_data[key]
    for key in intersect_set:
        has_dict_item = False
        value = base_data[key]
        if isinstance(value, list):
            p_list = base_data[key] if key in base_data else []
            d_list = compare_with_data[key] if key in compare_with_data else []
            keys_to_compare = next((test_key_item[key] for test_key_item in test_keys if key in test_key_item), None)
            changed_list = []
            if p_list and d_list:
                for p_list_item in p_list:
                    matched = False
                    has_diff = False
                    for d_list_item in d_list:
                        if isinstance(p_list_item, dict) and isinstance(d_list_item, dict):
                            if keys_to_compare:
                                key_matched_cnt = 0
                                test_keys_present_cnt = 0
                                common_keys = set(p_list_item).intersection(d_list_item)
                                for test_key in keys_to_compare:
                                    if test_key in common_keys:
                                        test_keys_present_cnt += 1
                                        if p_list_item[test_key] == d_list_item[test_key]:
                                            key_matched_cnt += 1
                                if key_matched_cnt and key_matched_cnt == test_keys_present_cnt:
                                    remaining_keys = [test_key_item for test_key_item in test_keys if key not in test_key_item]
                                    dict_diff = get_diff_dict(p_list_item, d_list_item, remaining_keys, is_skeleton)
                                    matched = True
                                    if dict_diff:
                                        has_diff = True
                                        for test_key in keys_to_compare:
                                            dict_diff.update({test_key: p_list_item[test_key]})
                                    break
                            else:
                                dict_diff = get_diff_dict(p_list_item, d_list_item, test_keys, is_skeleton)
                                if not dict_diff:
                                    matched = True
                                    break
                        elif p_list_item == d_list_item:
                            matched = True
                            break
                    if not matched:
                        if is_skeleton:
                            changed_list.append(p_list_item)
                        elif isinstance(p_list_item, dict):
                            val_dict = remove_empties(p_list_item)
                            if val_dict is not None:
                                changed_list.append(val_dict)
                        elif isinstance(p_list_item, list):
                            val_list = remove_empties_from_list(p_list_item)
                            if val_list is not None:
                                changed_list.append(val_list)
                        elif p_list_item is not None:
                            changed_list.append(p_list_item)
                    elif has_diff and dict_diff:
                        changed_list.append(dict_diff)
                if changed_list:
                    changed_dict.update({key: changed_list})
            elif p_list and (not d_list):
                changed_dict[key] = p_list
        elif isinstance(value, dict) and isinstance(compare_with_data[key], dict):
            dict_diff = get_diff_dict(base_data[key], compare_with_data[key], test_keys, is_skeleton)
            if dict_diff:
                changed_dict[key] = dict_diff
        elif value is not None:
            if not is_skeleton:
                if compare_with_data[key] != base_data[key]:
                    changed_dict[key] = base_data[key]
    return changed_dict
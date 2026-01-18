from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import json
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.kubernetes.core.plugins.module_utils.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def list_merge(last_applied, actual, desired, position):
    result = list()
    if position in STRATEGIC_MERGE_PATCH_KEYS and last_applied:
        patch_merge_key = STRATEGIC_MERGE_PATCH_KEYS[position]
        last_applied_dict = list_to_dict(last_applied, patch_merge_key, position)
        actual_dict = list_to_dict(actual, patch_merge_key, position)
        desired_dict = list_to_dict(desired, patch_merge_key, position)
        for key in desired_dict:
            if key not in actual_dict or key not in last_applied_dict:
                result.append(desired_dict[key])
            else:
                patch = merge(last_applied_dict[key], desired_dict[key], actual_dict[key], position)
                result.append(dict_merge(actual_dict[key], patch))
        for key in actual_dict:
            if key not in desired_dict and key not in last_applied_dict:
                result.append(actual_dict[key])
        return result
    else:
        return desired
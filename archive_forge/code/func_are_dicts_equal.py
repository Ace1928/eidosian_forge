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
def are_dicts_equal(option_name, existing_resource_dict, user_provided_dict, exclude_list, default_attribute_values):
    if not user_provided_dict:
        return is_attr_assigned_default(default_attribute_values, option_name, existing_resource_dict)
    if not existing_resource_dict and user_provided_dict:
        return False
    for sub_attr in existing_resource_dict:
        if sub_attr in user_provided_dict:
            if existing_resource_dict[sub_attr] != user_provided_dict[sub_attr]:
                _debug("Failed to match: Existing resource's attr {0} sub-attr {1} value is {2}, while user provided value is {3}".format(option_name, sub_attr, existing_resource_dict[sub_attr], user_provided_dict.get(sub_attr, None)))
                return False
        elif not should_dict_attr_be_excluded(option_name, sub_attr, exclude_list):
            default_value_for_dict_attr = default_attribute_values.get(option_name, None)
            if default_value_for_dict_attr:
                if not is_attr_assigned_default(default_value_for_dict_attr, sub_attr, existing_resource_dict[sub_attr]):
                    return False
            else:
                _debug("Consider as match: Existing resource's attr {0} sub-attr {1} value is {2}, while user didnot provide a value for it. The module author also has not provided a default value for itor marked it for exclusion. So ignoring this attribute during matching and continuing withother checks".format(option_name, sub_attr, existing_resource_dict[sub_attr]))
    return True
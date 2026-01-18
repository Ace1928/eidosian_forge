from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def get_dict_of_struct_follow(struct, filter_keys):
    if isinstance(struct, sdk.Struct):
        res = {}
        for key, value in struct.__dict__.items():
            if value is None:
                continue
            key = remove_underscore(key)
            if filter_keys is None or key in filter_keys:
                res[key] = get_dict_of_struct_follow(value, filter_keys)
        return res
    elif isinstance(struct, Enum) or isinstance(struct, datetime):
        return str(struct)
    elif isinstance(struct, list) or isinstance(struct, sdk.List):
        return [get_dict_of_struct_follow(i, filter_keys) for i in struct]
    return struct
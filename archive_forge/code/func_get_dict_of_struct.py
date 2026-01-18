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
def get_dict_of_struct(struct, connection=None, fetch_nested=False, attributes=None, filter_keys=None, follow=None):
    """
    Convert SDK Struct type into dictionary.
    """
    if follow:
        return get_dict_of_struct_follow(struct, filter_keys)
    res = {}

    def resolve_href(value):
        try:
            value = connection.follow_link(value)
        except sdk.Error:
            value = None
        nested_obj = dict(((attr, convert_value(getattr(value, attr))) for attr in attributes if getattr(value, attr, None) is not None))
        nested_obj['id'] = getattr(value, 'id', None)
        nested_obj['href'] = getattr(value, 'href', None)
        return nested_obj

    def convert_value(value):
        nested = False
        if isinstance(value, sdk.Struct):
            if not fetch_nested or not value.href:
                return get_dict_of_struct(value)
            return resolve_href(value)
        elif isinstance(value, Enum) or isinstance(value, datetime):
            return str(value)
        elif isinstance(value, list) or isinstance(value, sdk.List):
            if isinstance(value, sdk.List) and fetch_nested and value.href:
                try:
                    value = connection.follow_link(value)
                    nested = True
                except sdk.Error:
                    value = []
            ret = []
            for i in value:
                if isinstance(i, sdk.Struct):
                    if not nested and fetch_nested and i.href:
                        ret.append(resolve_href(i))
                    elif not nested:
                        ret.append(get_dict_of_struct(i))
                    else:
                        nested_obj = dict(((attr, convert_value(getattr(i, attr))) for attr in attributes if getattr(i, attr, None)))
                        nested_obj['id'] = getattr(i, 'id', None)
                        ret.append(nested_obj)
                elif isinstance(i, Enum):
                    ret.append(str(i))
                else:
                    ret.append(i)
            return ret
        else:
            return value
    if struct is not None:
        for key, value in struct.__dict__.items():
            if value is None:
                continue
            key = remove_underscore(key)
            if filter_keys is None:
                res[key] = convert_value(value)
            elif key in filter_keys:
                res[key] = convert_value(value)
    return res
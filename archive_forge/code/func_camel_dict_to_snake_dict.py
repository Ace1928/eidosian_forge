from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six.moves.collections_abc import MutableMapping
def camel_dict_to_snake_dict(camel_dict, reversible=False, ignore_list=()):
    """
    reversible allows two way conversion of a camelized dict
    such that snake_dict_to_camel_dict(camel_dict_to_snake_dict(x)) == x

    This is achieved through mapping e.g. HTTPEndpoint to h_t_t_p_endpoint
    where the default would be simply http_endpoint, which gets turned into
    HttpEndpoint if recamelized.

    ignore_list is used to avoid converting a sub-tree of a dict. This is
    particularly important for tags, where keys are case-sensitive. We convert
    the 'Tags' key but nothing below.
    """

    def value_is_list(camel_list):
        checked_list = []
        for item in camel_list:
            if isinstance(item, dict):
                checked_list.append(camel_dict_to_snake_dict(item, reversible))
            elif isinstance(item, list):
                checked_list.append(value_is_list(item))
            else:
                checked_list.append(item)
        return checked_list
    snake_dict = {}
    for k, v in camel_dict.items():
        if isinstance(v, dict) and k not in ignore_list:
            snake_dict[_camel_to_snake(k, reversible=reversible)] = camel_dict_to_snake_dict(v, reversible)
        elif isinstance(v, list) and k not in ignore_list:
            snake_dict[_camel_to_snake(k, reversible=reversible)] = value_is_list(v)
        else:
            snake_dict[_camel_to_snake(k, reversible=reversible)] = v
    return snake_dict
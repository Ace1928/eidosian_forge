from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_to_dict(obj, obj_type='standard'):
    """Generic function to convert a pyrax object to a dict

    obj_type values:
        standard
        clb
        server

    """
    instance = {}
    for key in dir(obj):
        value = getattr(obj, key)
        if obj_type == 'clb' and key == 'nodes':
            instance[key] = []
            for node in value:
                instance[key].append(rax_clb_node_to_dict(node))
        elif isinstance(value, list) and len(value) > 0 and (not isinstance(value[0], NON_CALLABLES)):
            instance[key] = []
            for item in value:
                instance[key].append(rax_to_dict(item))
        elif isinstance(value, NON_CALLABLES) and (not key.startswith('_')):
            if obj_type == 'server':
                if key == 'image':
                    if not value:
                        instance['rax_boot_source'] = 'volume'
                    else:
                        instance['rax_boot_source'] = 'local'
                key = rax_slugify(key)
            instance[key] = value
    if obj_type == 'server':
        for attr in ['id', 'accessIPv4', 'name', 'status']:
            instance[attr] = instance.get(rax_slugify(attr))
    return instance
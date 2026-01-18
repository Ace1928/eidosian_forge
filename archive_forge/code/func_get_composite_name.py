from __future__ import (absolute_import, division, print_function)
import os
from ansible import context
from ansible import constants as C
from ansible.collections.list import list_collections
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible.plugins import loader
from ansible.utils.display import Display
from ansible.utils.collection_loader._collection_finder import _get_collection_path
def get_composite_name(collection, name, path, depth):
    resolved_collection = collection
    if '.' not in name:
        resource_name = name
    else:
        if collection == 'ansible.legacy' and name.startswith('ansible.builtin.'):
            resolved_collection = 'ansible.builtin'
        resource_name = '.'.join(name.split(f'{resolved_collection}.')[1:])
    composite = [resolved_collection]
    if depth:
        composite.extend(path.split(os.path.sep)[depth * -1:])
    composite.append(to_native(resource_name))
    return '.'.join(composite)
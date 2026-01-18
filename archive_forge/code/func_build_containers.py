from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def build_containers(containers, vim_type, names, filters):
    filters = filters or []
    if vim_type:
        containers = filter_containers(containers, vim_type, names)
    new_containers = []
    for fil in filters:
        new_filters = None
        for k, v in fil.items():
            if k == 'resources':
                new_filters = v
            else:
                vim_type = getattr(vim, _snake_to_camel(k, True))
                names = v
                type_to_name_map[vim_type] = k.replace('_', ' ')
        new_containers.extend(build_containers(containers, vim_type, names, new_filters))
    if len(filters) > 0:
        return new_containers
    return containers
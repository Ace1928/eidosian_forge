from __future__ import absolute_import, division, print_function
import ssl
import atexit
import base64
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils._text import to_text, to_native
def filter_containers(containers, typ, filter_list):
    if len(filter_list) > 0:
        objs = []
        results = []
        found_filters = {}
        for container in containers:
            results.extend(get_contents(container, [typ]))
        for res in results:
            if res.propSet[0].val in filter_list:
                objs.append(res.obj)
                found_filters[res.propSet[0].val] = True
        for fil in filter_list:
            if fil not in found_filters:
                _handle_error('Unable to find %s %s' % (type_to_name_map[typ], fil))
        return objs
    return containers
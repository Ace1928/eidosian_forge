from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb_canonical_map import base_interfaces
def ciscosmb_merge_dicts(a, b, path=None):
    """merges b into a"""
    if path is None:
        path = []
    if not bool(b):
        return a
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                ciscosmb_merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
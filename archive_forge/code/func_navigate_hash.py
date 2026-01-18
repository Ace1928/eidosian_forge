from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
def navigate_hash(source, path, default=None):
    if not source:
        return None
    key = path[0]
    path = path[1:]
    if key not in source:
        return default
    result = source[key]
    if path:
        return navigate_hash(result, path, default)
    return result
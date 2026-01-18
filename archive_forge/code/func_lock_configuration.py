from __future__ import absolute_import, division, print_function
import collections
import json
from contextlib import contextmanager
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
def lock_configuration(module):
    conn = get_connection(module)
    try:
        response = conn.lock()
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    return response
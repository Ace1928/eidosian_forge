from __future__ import absolute_import, division, print_function
import json
import re
from difflib import Differ
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
def discard_config(module):
    conn = get_connection(module)
    try:
        if is_netconf(module):
            conn.discard_changes(remove_ns=True)
        else:
            conn.discard_changes()
    except ConnectionError as exc:
        module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
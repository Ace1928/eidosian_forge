from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def save_module_context(self, module_key, module_context):
    try:
        self._connection.save_module_context(module_key, module_context)
    except ConnectionError as exc:
        self._module.fail_json(msg=to_text(exc, errors='surrogate_then_replace'))
    return None
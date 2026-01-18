from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def check_legacy_fortiosapi(module):
    legacy_schemas = ['host', 'username', 'password', 'ssl_verify', 'https']
    legacy_params = []
    for param in legacy_schemas:
        if param in module.params:
            legacy_params.append(param)
    if len(legacy_params):
        error_message = 'Legacy fortiosapi parameters %s detected, please use HTTPAPI instead!' % str(legacy_params)
        module.fail_json(msg=error_message)
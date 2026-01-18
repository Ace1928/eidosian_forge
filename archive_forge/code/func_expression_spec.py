from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def expression_spec():
    return dict(key=dict(type='str', required=True, no_log=False), operator=dict(type='str', choices=['not_in', 'in', 'equals', 'not_equals', 'has_key', 'does_not_have_key'], required=True), value=dict(type='str'))
from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def netflow_spec():
    return dict(name=dict(type='str', required=True), active_flow_timeout=dict(type='int'), idle_flow_timeout=dict(type='int'), sampling_rate=dict(type='int'))
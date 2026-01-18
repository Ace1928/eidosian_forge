from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def route_control_profile_spec():
    return dict(profile=dict(type='str', required=True), l3out=dict(type='str'), direction=dict(type='str', required=True), tenant=dict(type='str', required=True))
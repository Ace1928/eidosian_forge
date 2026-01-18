from __future__ import (absolute_import, division, print_function)
import json
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
def online_argument_spec():
    return dict(api_token=dict(required=True, fallback=(env_fallback, ['ONLINE_TOKEN', 'ONLINE_API_KEY', 'ONLINE_OAUTH_TOKEN', 'ONLINE_API_TOKEN']), no_log=True, aliases=['oauth_token']), api_url=dict(fallback=(env_fallback, ['ONLINE_API_URL']), default='https://api.online.net', aliases=['base_url']), api_timeout=dict(type='int', default=30, aliases=['timeout']), validate_certs=dict(default=True, type='bool'))
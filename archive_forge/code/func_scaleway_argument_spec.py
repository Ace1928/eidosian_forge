from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def scaleway_argument_spec():
    return dict(api_token=dict(required=True, fallback=(env_fallback, ['SCW_TOKEN', 'SCW_API_KEY', 'SCW_OAUTH_TOKEN', 'SCW_API_TOKEN']), no_log=True, aliases=['oauth_token']), api_url=dict(fallback=(env_fallback, ['SCW_API_URL']), default='https://api.scaleway.com', aliases=['base_url']), api_timeout=dict(type='int', default=30, aliases=['timeout']), query_parameters=dict(type='dict', default={}), validate_certs=dict(default=True, type='bool'))
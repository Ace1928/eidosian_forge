from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def cs_argument_spec():
    return dict(api_key=dict(type='str', fallback=(env_fallback, ['CLOUDSTACK_KEY']), required=True, no_log=False), api_secret=dict(type='str', fallback=(env_fallback, ['CLOUDSTACK_SECRET']), required=True, no_log=True), api_url=dict(type='str', fallback=(env_fallback, ['CLOUDSTACK_ENDPOINT']), required=True), api_http_method=dict(type='str', fallback=(env_fallback, ['CLOUDSTACK_METHOD']), choices=['get', 'post'], default='get'), api_timeout=dict(type='int', fallback=(env_fallback, ['CLOUDSTACK_TIMEOUT']), default=10), api_verify_ssl_cert=dict(type='str', fallback=(env_fallback, ['CLOUDSTACK_VERIFY'])))
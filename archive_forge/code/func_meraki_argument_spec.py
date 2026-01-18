from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def meraki_argument_spec():
    return dict(auth_key=dict(type='str', no_log=True, fallback=(env_fallback, ['MERAKI_KEY']), required=True), host=dict(type='str', default='api.meraki.com'), use_proxy=dict(type='bool', default=False), use_https=dict(type='bool', default=True), validate_certs=dict(type='bool', default=True), output_format=dict(type='str', choices=['camelcase', 'snakecase'], default='snakecase', fallback=(env_fallback, ['ANSIBLE_MERAKI_FORMAT'])), output_level=dict(type='str', default='normal', choices=['normal', 'debug']), timeout=dict(type='int', default=30), org_name=dict(type='str', aliases=['organization']), org_id=dict(type='str'), rate_limit_retry_time=dict(type='int', default=165), internal_error_retry_time=dict(type='int', default=60))
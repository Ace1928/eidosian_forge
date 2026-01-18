from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
@staticmethod
def digital_ocean_argument_spec():
    return dict(baseurl=dict(type='str', required=False, default='https://api.digitalocean.com/v2'), validate_certs=dict(type='bool', required=False, default=True), oauth_token=dict(no_log=True, fallback=(env_fallback, ['DO_API_TOKEN', 'DO_API_KEY', 'DO_OAUTH_TOKEN', 'OAUTH_TOKEN']), required=False, aliases=['api_token']), timeout=dict(type='int', default=30))
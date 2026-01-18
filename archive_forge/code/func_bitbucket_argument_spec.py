from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url, basic_auth_header
@staticmethod
def bitbucket_argument_spec():
    return dict(client_id=dict(type='str', fallback=(env_fallback, ['BITBUCKET_CLIENT_ID'])), client_secret=dict(type='str', no_log=True, fallback=(env_fallback, ['BITBUCKET_CLIENT_SECRET'])), user=dict(type='str', aliases=['username'], fallback=(env_fallback, ['BITBUCKET_USERNAME'])), password=dict(type='str', no_log=True, fallback=(env_fallback, ['BITBUCKET_PASSWORD'])))
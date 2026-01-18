from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url, basic_auth_header
@staticmethod
def bitbucket_required_one_of():
    return [['client_id', 'client_secret', 'user', 'password']]
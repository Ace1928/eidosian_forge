from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
def cloudscale_argument_spec():
    return dict(api_url=dict(type='str', fallback=(env_fallback, ['CLOUDSCALE_API_URL']), default='https://api.cloudscale.ch/v1'), api_token=dict(type='str', fallback=(env_fallback, ['CLOUDSCALE_API_TOKEN']), no_log=True, required=True), api_timeout=dict(type='int', fallback=(env_fallback, ['CLOUDSCALE_API_TIMEOUT']), default=45))
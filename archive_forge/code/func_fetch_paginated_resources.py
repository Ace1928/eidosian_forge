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
def fetch_paginated_resources(self, resource_key, **pagination_kwargs):
    response = self.get(path=self.api_path, params=pagination_kwargs)
    status_code = response.status_code
    if not response.ok:
        self.module.fail_json(msg='Error getting {0} [{1}: {2}]'.format(resource_key, response.status_code, response.json['message']))
    return response.json[resource_key]
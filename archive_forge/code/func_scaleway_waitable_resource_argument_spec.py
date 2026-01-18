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
def scaleway_waitable_resource_argument_spec():
    return dict(wait=dict(type='bool', default=True), wait_timeout=dict(type='int', default=300), wait_sleep_time=dict(type='int', default=3))
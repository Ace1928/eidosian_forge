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
@staticmethod
def get_user_agent_string(module):
    return 'ansible %s Python %s' % (module.ansible_version, sys.version.split(' ', 1)[0])
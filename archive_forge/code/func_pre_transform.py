from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
def pre_transform(self, resource):
    return resource
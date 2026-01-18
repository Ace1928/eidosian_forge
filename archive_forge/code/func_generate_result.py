from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def generate_result(changed, actions=None, modify=None, response=None, extra_responses=None):
    result = dict(changed=changed)
    if response is not None:
        result['response'] = response
    if modify:
        result['modify'] = modify
    if actions:
        result['actions'] = actions
    if extra_responses:
        result.update(extra_responses)
    return result
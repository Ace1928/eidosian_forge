from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def is_zapi_write_access_error(message):
    """ return True if it is a write access error """
    if isinstance(message, str) and message.startswith('Insufficient privileges:'):
        return 'does not have write access' in message
    return False
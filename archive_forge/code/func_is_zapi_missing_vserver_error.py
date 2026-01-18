from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def is_zapi_missing_vserver_error(message):
    """ return True if it is a missing vserver error """
    return isinstance(message, str) and message in ('Vserver API missing vserver parameter.', 'Specified vserver not found')
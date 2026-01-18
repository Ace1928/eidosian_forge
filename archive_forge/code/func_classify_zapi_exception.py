from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def classify_zapi_exception(error):
    """ return type of error """
    try:
        err_code = int(error.code)
    except (AttributeError, ValueError):
        err_code = 0
    try:
        err_msg = error.message
    except AttributeError:
        err_msg = ''
    if err_code == 13005 and err_msg.startswith('Unable to find API:') and ('data vserver' in err_msg):
        return ('missing_vserver_api_error', 'Most likely running a cluster level API as vserver: %s' % to_native(error))
    if err_code == 13001 and err_msg.startswith("RPC: Couldn't make connection"):
        return ('rpc_error', to_native(error))
    return ('other_error', to_native(error))
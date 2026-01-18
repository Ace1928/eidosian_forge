from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_system_res_id(idrac):
    res_id = SYSTEM_ID
    error_msg = ''
    try:
        resp = idrac.invoke_request(SYSTEMS_URI, 'GET')
    except HTTPError:
        error_msg = 'Unable to complete the request because the resource URI does not exist or is not implemented.'
    else:
        member = resp.json_data.get('Members')
        res_uri = member[0].get('@odata.id')
        res_id = res_uri.split('/')[-1]
    return (res_id, error_msg)
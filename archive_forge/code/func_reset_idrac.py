from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def reset_idrac(idrac_restobj, wait_time_sec=300, res_id=MANAGER_ID, interval=30):
    track_failed = True
    reset_msg = 'iDRAC reset triggered successfully.'
    try:
        idrac_restobj.invoke_request(IDRAC_RESET_URI.format(res_id=res_id), 'POST', data={'ResetType': 'GracefulRestart'})
        if wait_time_sec:
            track_failed, reset_msg = wait_after_idrac_reset(idrac_restobj, wait_time_sec, interval)
        reset = True
    except Exception:
        reset = False
        reset_msg = RESET_FAIL
    return (reset, track_failed, reset_msg)
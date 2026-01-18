import re
import sys
import warnings
def check_status(status):
    status = check_string_type(status, 'Status')
    status_code = status.split(None, 1)[0]
    assert_(len(status_code) == 3, 'Status codes must be three characters: %r' % status_code)
    status_int = int(status_code)
    assert_(status_int >= 100, 'Status code is invalid: %r' % status_int)
    if len(status) < 4 or status[3] != ' ':
        warnings.warn('The status string (%r) should be a three-digit integer followed by a single space and a status explanation' % status, WSGIWarning)
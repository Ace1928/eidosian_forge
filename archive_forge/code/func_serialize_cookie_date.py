import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def serialize_cookie_date(v):
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v
    elif isinstance(v, text_type):
        return v.encode('ascii')
    elif isinstance(v, int):
        v = timedelta(seconds=v)
    if isinstance(v, timedelta):
        v = datetime.utcnow() + v
    if isinstance(v, (datetime, date)):
        v = v.timetuple()
    r = time.strftime('%%s, %d-%%s-%Y %H:%M:%S GMT', v)
    return bytes_(r % (weekdays[v[6]], months[v[1]]), 'ascii')
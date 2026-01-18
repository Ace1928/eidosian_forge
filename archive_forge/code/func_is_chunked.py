import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
def is_chunked(self):
    if self.response_length is not None:
        return False
    elif self.req.version <= (1, 0):
        return False
    elif self.req.method == 'HEAD':
        return False
    elif self.status_code in (204, 304):
        return False
    return True
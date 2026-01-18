from http.cookies import SimpleCookie
import time
import random
import os
import datetime
import threading
import tempfile
from paste import wsgilib
from paste import request
def session_start_response(status, headers, exc_info=None):
    if not session_factory.created:
        remember_headers[:] = [status, headers]
        return start_response(status, headers)
    headers.append(session_factory.set_cookie_header())
    return start_response(status, headers, exc_info)
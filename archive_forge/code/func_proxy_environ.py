import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
def proxy_environ(req):
    info = req.proxy_protocol_info
    if not info:
        return {}
    return {'PROXY_PROTOCOL': info['proxy_protocol'], 'REMOTE_ADDR': info['client_addr'], 'REMOTE_PORT': str(info['client_port']), 'PROXY_ADDR': info['proxy_addr'], 'PROXY_PORT': str(info['proxy_port'])}
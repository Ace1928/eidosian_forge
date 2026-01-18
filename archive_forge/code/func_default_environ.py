import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
def default_environ(req, sock, cfg):
    env = base_environ(cfg)
    env.update({'wsgi.input': req.body, 'gunicorn.socket': sock, 'REQUEST_METHOD': req.method, 'QUERY_STRING': req.query, 'RAW_URI': req.uri, 'SERVER_PROTOCOL': 'HTTP/%s' % '.'.join([str(v) for v in req.version])})
    return env
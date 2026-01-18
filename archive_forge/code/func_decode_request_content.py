from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
def decode_request_content(self, data):
    encoding = self.headers.get('content-encoding', 'identity').lower()
    if encoding == 'identity':
        return data
    if encoding == 'gzip':
        try:
            return gzip_decode(data)
        except NotImplementedError:
            self.send_response(501, 'encoding %r not supported' % encoding)
        except ValueError:
            self.send_response(400, 'error decoding gzip content')
    else:
        self.send_response(501, 'encoding %r not supported' % encoding)
    self.send_header('Content-length', '0')
    self.end_headers()
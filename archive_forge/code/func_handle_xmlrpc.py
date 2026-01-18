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
def handle_xmlrpc(self, request_text):
    """Handle a single XML-RPC request"""
    response = self._marshaled_dispatch(request_text)
    print('Content-Type: text/xml')
    print('Content-Length: %d' % len(response))
    print()
    sys.stdout.flush()
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()
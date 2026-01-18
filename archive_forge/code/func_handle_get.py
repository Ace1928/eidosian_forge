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
def handle_get(self):
    """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """
    response = self.generate_html_documentation().encode('utf-8')
    print('Content-Type: text/html')
    print('Content-Length: %d' % len(response))
    print()
    sys.stdout.flush()
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()
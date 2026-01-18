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
def report_404(self):
    self.send_response(404)
    response = b'No such page'
    self.send_header('Content-type', 'text/plain')
    self.send_header('Content-length', str(len(response)))
    self.end_headers()
    self.wfile.write(response)
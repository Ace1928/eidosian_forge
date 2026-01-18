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
def set_server_documentation(self, server_documentation):
    """Set the documentation string for the entire server."""
    self.server_documentation = server_documentation
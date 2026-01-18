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
def is_rpc_path_valid(self):
    if self.rpc_paths:
        return self.path in self.rpc_paths
    else:
        return True
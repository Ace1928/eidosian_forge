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
def system_listMethods(self):
    """system.listMethods() => ['add', 'subtract', 'multiple']

        Returns a list of the methods supported by the server."""
    methods = set(self.funcs.keys())
    if self.instance is not None:
        if hasattr(self.instance, '_listMethods'):
            methods |= set(self.instance._listMethods())
        elif not hasattr(self.instance, '_dispatch'):
            methods |= set(list_public_methods(self.instance))
    return sorted(methods)
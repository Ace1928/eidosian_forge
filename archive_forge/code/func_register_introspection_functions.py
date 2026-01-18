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
def register_introspection_functions(self):
    """Registers the XML-RPC introspection methods in the system
        namespace.

        see http://xmlrpc.usefulinc.com/doc/reserved.html
        """
    self.funcs.update({'system.listMethods': self.system_listMethods, 'system.methodSignature': self.system_methodSignature, 'system.methodHelp': self.system_methodHelp})
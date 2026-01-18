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
def generate_html_documentation(self):
    """generate_html_documentation() => html documentation for the server

        Generates HTML documentation for the server using introspection for
        installed functions and instances that do not implement the
        _dispatch method. Alternatively, instances can choose to implement
        the _get_method_argstring(method_name) method to provide the
        argument string used in the documentation and the
        _methodHelp(method_name) method to provide the help text used
        in the documentation."""
    methods = {}
    for method_name in self.system_listMethods():
        if method_name in self.funcs:
            method = self.funcs[method_name]
        elif self.instance is not None:
            method_info = [None, None]
            if hasattr(self.instance, '_get_method_argstring'):
                method_info[0] = self.instance._get_method_argstring(method_name)
            if hasattr(self.instance, '_methodHelp'):
                method_info[1] = self.instance._methodHelp(method_name)
            method_info = tuple(method_info)
            if method_info != (None, None):
                method = method_info
            elif not hasattr(self.instance, '_dispatch'):
                try:
                    method = resolve_dotted_attribute(self.instance, method_name)
                except AttributeError:
                    method = method_info
            else:
                method = method_info
        else:
            assert 0, 'Could not find method in self.functions and no instance installed'
        methods[method_name] = method
    documenter = ServerHTMLDoc()
    documentation = documenter.docserver(self.server_name, self.server_documentation, methods)
    return documenter.page(html.escape(self.server_title), documentation)
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
def docroutine(self, object, name, mod=None, funcs={}, classes={}, methods={}, cl=None):
    """Produce HTML documentation for a function or method object."""
    anchor = (cl and cl.__name__ or '') + '-' + name
    note = ''
    title = '<a name="%s"><strong>%s</strong></a>' % (self.escape(anchor), self.escape(name))
    if callable(object):
        argspec = str(signature(object))
    else:
        argspec = '(...)'
    if isinstance(object, tuple):
        argspec = object[0] or argspec
        docstring = object[1] or ''
    else:
        docstring = pydoc.getdoc(object)
    decl = title + argspec + (note and self.grey('<font face="helvetica, arial">%s</font>' % note))
    doc = self.markup(docstring, self.preformat, funcs, classes, methods)
    doc = doc and '<dd><tt>%s</tt></dd>' % doc
    return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)
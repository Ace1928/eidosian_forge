import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
def no_escape(value):
    if value is None:
        return ''
    if not isinstance(value, text_type):
        if hasattr(value, '__unicode__'):
            value = value.__unicode__()
        if isinstance(value, bytes):
            value = text_(value, 'utf-8')
        else:
            value = text_type(value)
    return value
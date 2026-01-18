import json
from string import Template
import re
import sys
from webob.acceptparse import create_accept_header
from webob.compat import (
from webob.request import Request
from webob.response import Response
from webob.util import html_escape
def plain_body(self, environ):
    body = self._make_body(environ, no_escape)
    body = strip_tags(body)
    return self.plain_template_obj.substitute(status=self.status, title=self.title, body=body)
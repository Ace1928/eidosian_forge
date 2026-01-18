import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def render_response(body=None, status=None, headers=None, method=None):
    """Form a WSGI response."""
    if headers is None:
        headers = []
    else:
        headers = list(headers)
    headers.append(('Vary', 'X-Auth-Token'))
    if body is None:
        body = b''
        status = status or (http.client.NO_CONTENT, http.client.responses[http.client.NO_CONTENT])
    else:
        content_types = [v for h, v in headers if h == 'Content-Type']
        if content_types:
            content_type = content_types[0]
        else:
            content_type = None
        if content_type is None or content_type in JSON_ENCODE_CONTENT_TYPES:
            body = jsonutils.dump_as_bytes(body, cls=utils.SmarterEncoder)
            if content_type is None:
                headers.append(('Content-Type', 'application/json'))
        status = status or (http.client.OK, http.client.responses[http.client.OK])

    def _convert_to_str(headers):
        str_headers = []
        for header in headers:
            str_header = []
            for value in header:
                if not isinstance(value, str):
                    str_header.append(str(value))
                else:
                    str_header.append(value)
            str_headers.append(tuple(str_header))
        return str_headers
    headers = _convert_to_str(headers)
    resp = webob.Response(body=body, status='%d %s' % status, headerlist=headers, charset='utf-8')
    if method and method.upper() == 'HEAD':
        stored_headers = resp.headers.copy()
        resp.body = b''
        for header, value in stored_headers.items():
            resp.headers[header] = value
    return resp
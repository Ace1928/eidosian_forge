import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def response_headers(headers=None, debug=False):
    """Set headers on the response."""
    if debug:
        cherrypy.log('Setting response headers: %s' % repr(headers), 'TOOLS.RESPONSE_HEADERS')
    for name, value in headers or []:
        cherrypy.serving.response.headers[name] = value
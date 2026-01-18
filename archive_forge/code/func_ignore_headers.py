import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def ignore_headers(headers=('Range',), debug=False):
    """Delete request headers whose field names are included in 'headers'.

    This is a useful tool for working behind certain HTTP servers;
    for example, Apache duplicates the work that CP does for 'Range'
    headers, and will doubly-truncate the response.
    """
    request = cherrypy.serving.request
    for name in headers:
        if name in request.headers:
            if debug:
                cherrypy.log('Ignoring request header %r' % name, 'TOOLS.IGNORE_HEADERS')
            del request.headers[name]
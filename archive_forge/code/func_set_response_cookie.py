import sys
import datetime
import os
import time
import threading
import binascii
import pickle
import zc.lockfile
import cherrypy
from cherrypy.lib import httputil
from cherrypy.lib import locking
from cherrypy.lib import is_iterator
def set_response_cookie(path=None, path_header=None, name='session_id', timeout=60, domain=None, secure=False, httponly=False):
    """Set a response cookie for the client.

    path
        the 'path' value to stick in the response cookie metadata.

    path_header
        if 'path' is None (the default), then the response
        cookie 'path' will be pulled from request.headers[path_header].

    name
        the name of the cookie.

    timeout
        the expiration timeout for the cookie. If 0 or other boolean
        False, no 'expires' param will be set, and the cookie will be a
        "session cookie" which expires when the browser is closed.

    domain
        the cookie domain.

    secure
        if False (the default) the cookie 'secure' value will not
        be set. If True, the cookie 'secure' value will be set (to 1).

    httponly
        If False (the default) the cookie 'httponly' value will not be set.
        If True, the cookie 'httponly' value will be set (to 1).

    """
    cookie = cherrypy.serving.response.cookie
    cookie[name] = cherrypy.serving.session.id
    cookie[name]['path'] = path or cherrypy.serving.request.headers.get(path_header) or '/'
    if timeout:
        cookie[name]['max-age'] = timeout * 60
        _add_MSIE_max_age_workaround(cookie[name], timeout)
    if domain is not None:
        cookie[name]['domain'] = domain
    if secure:
        cookie[name]['secure'] = 1
    if httponly:
        if not cookie[name].isReservedKey('httponly'):
            raise ValueError('The httponly cookie token is not supported.')
        cookie[name]['httponly'] = 1
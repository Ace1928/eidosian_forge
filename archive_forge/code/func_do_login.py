import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def do_login(self, username, password, from_page='..', **kwargs):
    """Login. May raise redirect, or return True if request handled."""
    response = cherrypy.serving.response
    error_msg = self.check_username_and_password(username, password)
    if error_msg:
        body = self.login_screen(from_page, username, error_msg)
        response.body = body
        if 'Content-Length' in response.headers:
            del response.headers['Content-Length']
        return True
    else:
        cherrypy.serving.request.login = username
        cherrypy.session[self.session_key] = username
        self.on_login(username)
        raise cherrypy.HTTPRedirect(from_page or '/')
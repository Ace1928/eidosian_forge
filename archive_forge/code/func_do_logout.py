import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def do_logout(self, from_page='..', **kwargs):
    """Logout. May raise redirect, or return True if request handled."""
    sess = cherrypy.session
    username = sess.get(self.session_key)
    sess[self.session_key] = None
    if username:
        cherrypy.serving.request.login = None
        self.on_logout(username)
    raise cherrypy.HTTPRedirect(from_page)
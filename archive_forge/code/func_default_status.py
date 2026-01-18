import io
import contextlib
import urllib.parse
from sys import exc_info as _exc_info
from traceback import format_exception as _format_exception
from xml.sax import saxutils
import html
from more_itertools import always_iterable
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy._cpcompat import tonative
from cherrypy._helper import classproperty
from cherrypy.lib import httputil as _httputil
@classproperty
def default_status(cls):
    """
        The default redirect status for the request.

        RFC 2616 indicates a 301 response code fits our goal; however,
        browser support for 301 is quite messy. Use 302/303 instead. See
        http://www.alanflavell.org.uk/www/post-redirect.html
        """
    return 303 if cherrypy.serving.request.protocol >= (1, 1) else 302
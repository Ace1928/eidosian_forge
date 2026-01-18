import logging
import re
from hashlib import md5
import urllib.parse
import cherrypy
from cherrypy._cpcompat import text_or_bytes
from cherrypy.lib import httputil as _httputil
from cherrypy.lib import is_iterator
def convert_params(exception=ValueError, error=400):
    """Convert request params based on function annotations.

    This function also processes errors that are subclasses of ``exception``.

    :param BaseException exception: Exception class to catch.
    :type exception: BaseException

    :param error: The HTTP status code to return to the client on failure.
    :type error: int
    """
    request = cherrypy.serving.request
    types = request.handler.callable.__annotations__
    with cherrypy.HTTPError.handle(exception, error):
        for key in set(types).intersection(request.params):
            request.params[key] = types[key](request.params[key])
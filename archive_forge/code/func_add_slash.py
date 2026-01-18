import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
def add_slash(self, environ, start_response):
    """
        This happens when you try to get to a directory
        without a trailing /
        """
    url = request.construct_url(environ, with_query_string=False)
    url += '/'
    if environ.get('QUERY_STRING'):
        url += '?' + environ['QUERY_STRING']
    exc = httpexceptions.HTTPMovedPermanently('The resource has moved to %s - you should be redirected automatically.' % url, headers=[('location', url)])
    return exc.wsgi_application(environ, start_response)
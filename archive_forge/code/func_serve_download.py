import mimetypes
import os
import platform
import re
import stat
import unicodedata
import urllib.parse
from email.generator import _make_boundary as make_boundary
from io import UnsupportedOperation
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import cptools, file_generator_limited, httputil
def serve_download(path, name=None):
    """Serve 'path' as an application/x-download attachment."""
    return serve_file(path, 'application/x-download', 'attachment', name)
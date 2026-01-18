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
def serve_file(path, content_type=None, disposition=None, name=None, debug=False):
    """Set status, headers, and body in order to serve the given path.

    The Content-Type header will be set to the content_type arg, if provided.
    If not provided, the Content-Type will be guessed by the file extension
    of the 'path' argument.

    If disposition is not None, the Content-Disposition header will be set
    to "<disposition>; filename=<name>; filename*=utf-8''<name>"
    as described in :rfc:`6266#appendix-D`.
    If name is None, it will be set to the basename of path.
    If disposition is None, no Content-Disposition header will be written.
    """
    response = cherrypy.serving.response
    if not os.path.isabs(path):
        msg = "'%s' is not an absolute path." % path
        if debug:
            cherrypy.log(msg, 'TOOLS.STATICFILE')
        raise ValueError(msg)
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        if debug:
            cherrypy.log('os.stat(%r) failed' % path, 'TOOLS.STATIC')
        raise cherrypy.NotFound()
    if stat.S_ISDIR(st.st_mode):
        if debug:
            cherrypy.log('%r is a directory' % path, 'TOOLS.STATIC')
        raise cherrypy.NotFound()
    response.headers['Last-Modified'] = httputil.HTTPDate(st.st_mtime)
    cptools.validate_since()
    if content_type is None:
        ext = ''
        i = path.rfind('.')
        if i != -1:
            ext = path[i:].lower()
        content_type = mimetypes.types_map.get(ext, None)
    if content_type is not None:
        response.headers['Content-Type'] = content_type
    if debug:
        cherrypy.log('Content-Type: %r' % content_type, 'TOOLS.STATIC')
    cd = None
    if disposition is not None:
        if name is None:
            name = os.path.basename(path)
        cd = _make_content_disposition(disposition, name)
        response.headers['Content-Disposition'] = cd
    if debug:
        cherrypy.log('Content-Disposition: %r' % cd, 'TOOLS.STATIC')
    content_length = st.st_size
    fileobj = open(path, 'rb')
    return _serve_fileobj(fileobj, content_type, content_length, debug=debug)
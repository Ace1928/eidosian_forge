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
def serve_fileobj(fileobj, content_type=None, disposition=None, name=None, debug=False):
    """Set status, headers, and body in order to serve the given file object.

    The Content-Type header will be set to the content_type arg, if provided.

    If disposition is not None, the Content-Disposition header will be set
    to "<disposition>; filename=<name>; filename*=utf-8''<name>"
    as described in :rfc:`6266#appendix-D`.
    If name is None, 'filename' will not be set.
    If disposition is None, no Content-Disposition header will be written.

    CAUTION: If the request contains a 'Range' header, one or more seek()s will
    be performed on the file object.  This may cause undesired behavior if
    the file object is not seekable.  It could also produce undesired results
    if the caller set the read position of the file object prior to calling
    serve_fileobj(), expecting that the data would be served starting from that
    position.
    """
    response = cherrypy.serving.response
    try:
        st = os.fstat(fileobj.fileno())
    except AttributeError:
        if debug:
            cherrypy.log('os has no fstat attribute', 'TOOLS.STATIC')
        content_length = None
    except UnsupportedOperation:
        content_length = None
    else:
        response.headers['Last-Modified'] = httputil.HTTPDate(st.st_mtime)
        cptools.validate_since()
        content_length = st.st_size
    if content_type is not None:
        response.headers['Content-Type'] = content_type
    if debug:
        cherrypy.log('Content-Type: %r' % content_type, 'TOOLS.STATIC')
    cd = None
    if disposition is not None:
        if name is None:
            cd = disposition
        else:
            cd = _make_content_disposition(disposition, name)
        response.headers['Content-Disposition'] = cd
    if debug:
        cherrypy.log('Content-Disposition: %r' % cd, 'TOOLS.STATIC')
    return _serve_fileobj(fileobj, content_type, content_length, debug=debug)
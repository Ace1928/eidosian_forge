import warnings
import sys
from urllib import parse as urlparse
from paste.recursive import ForwardRequestException, RecursiveMiddleware, RecursionLoop
from paste.util import converters
from paste.response import replace_header
def make_errordocument(app, global_conf, **kw):
    """
    Paste Deploy entry point to create a error document wrapper.

    Use like::

        [filter-app:main]
        use = egg:Paste#errordocument
        next = real-app
        500 = /lib/msg/500.html
        404 = /lib/msg/404.html
    """
    map = {}
    for status, redir_loc in kw.items():
        try:
            status = int(status)
        except ValueError:
            raise ValueError('Bad status code: %r' % status)
        map[status] = redir_loc
    forwarder = forward(app, map)
    return forwarder
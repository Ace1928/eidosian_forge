import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
def downgrade_wsgi_ux_to_1x(environ):
    """Return a new environ dict for WSGI 1.x from the given WSGI u.x environ.
    """
    env1x = {}
    url_encoding = environ[ntou('wsgi.url_encoding')]
    for k, v in environ.copy().items():
        if k in [ntou('PATH_INFO'), ntou('SCRIPT_NAME'), ntou('QUERY_STRING')]:
            v = v.encode(url_encoding)
        elif isinstance(v, str):
            v = v.encode('ISO-8859-1')
        env1x[k.encode('ISO-8859-1')] = v
    return env1x
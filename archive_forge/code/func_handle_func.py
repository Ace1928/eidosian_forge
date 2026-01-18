import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
@expose
def handle_func(*a, **kw):
    handled = self.callable(*args, **self._merged_args(kwargs))
    if not handled:
        raise cherrypy.NotFound()
    return cherrypy.serving.response.body
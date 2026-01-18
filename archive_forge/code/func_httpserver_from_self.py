import cherrypy
from cherrypy.lib.reprconf import attributes
from cherrypy._cpcompat import text_or_bytes
from cherrypy.process.servers import ServerAdapter
def httpserver_from_self(self, httpserver=None):
    """Return a (httpserver, bind_addr) pair based on self attributes."""
    if httpserver is None:
        httpserver = self.instance
    if httpserver is None:
        from cherrypy import _cpwsgi_server
        httpserver = _cpwsgi_server.CPWSGIServer(self)
    if isinstance(httpserver, text_or_bytes):
        httpserver = attributes(httpserver)(self)
    return (httpserver, self.bind_addr)
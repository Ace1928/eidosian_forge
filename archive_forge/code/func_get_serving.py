import os
import cherrypy
from cherrypy import _cpconfig, _cplogging, _cprequest, _cpwsgi, tools
from cherrypy.lib import httputil, reprconf
def get_serving(self, local, remote, scheme, sproto):
    """Create and return a Request and Response object."""
    req = self.request_class(local, remote, scheme, sproto)
    req.app = self
    for name, toolbox in self.toolboxes.items():
        req.namespaces[name] = toolbox
    resp = self.response_class()
    cherrypy.serving.load(req, resp)
    cherrypy.engine.publish('acquire_thread')
    cherrypy.engine.publish('before_request')
    return (req, resp)
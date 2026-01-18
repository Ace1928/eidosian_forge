import cherrypy
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.proxy.base': None})
def base_no_base(self):
    return cherrypy.request.base
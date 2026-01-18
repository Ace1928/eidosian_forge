import cherrypy
from cherrypy.test import helper
@cherrypy.expose
def pageurl(self):
    return self.thisnewpage
import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
@cherrypy.expose
def confvalue(self):
    return cherrypy.request.config.get('user')
import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def notExposed(self):
    return 'not exposed'
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
def newurl(self):
    return "Browse to <a href='%s'>this page</a>." % cherrypy.url('/this/new/page')
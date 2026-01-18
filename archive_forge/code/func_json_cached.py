import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
@cherrypy.expose
@json_out
@cherrypy.config(**{'tools.caching.on': True})
def json_cached(self):
    return 'hello there'
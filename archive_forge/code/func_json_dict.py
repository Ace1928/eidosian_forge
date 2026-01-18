import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
@cherrypy.expose
@json_out
def json_dict(self):
    return {'answer': 42}
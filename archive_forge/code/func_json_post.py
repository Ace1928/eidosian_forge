import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
@cherrypy.expose
@json_in
def json_post(self):
    if cherrypy.request.json == [13, 'c']:
        return 'ok'
    else:
        return 'nok'
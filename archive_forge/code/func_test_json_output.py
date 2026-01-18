import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
def test_json_output(self):
    if json is None:
        self.skip('json not found ')
        return
    self.getPage('/plain')
    self.assertBody('hello')
    self.getPage('/json_string')
    self.assertBody('"hello"')
    self.getPage('/json_list')
    self.assertBody('["a", "b", 42]')
    self.getPage('/json_dict')
    self.assertBody('{"answer": 42}')
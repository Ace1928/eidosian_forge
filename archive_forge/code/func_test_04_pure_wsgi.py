import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def test_04_pure_wsgi(self):
    if not cherrypy.server.using_wsgi:
        return self.skip('skipped (not using WSGI)... ')
    self.getPage('/hosted/app1')
    self.assertHeader('Content-Type', 'text/plain')
    self.assertInBody(self.wsgi_output)
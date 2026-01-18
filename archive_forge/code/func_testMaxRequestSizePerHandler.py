import os
import cherrypy
from cherrypy.test import helper
def testMaxRequestSizePerHandler(self):
    if getattr(cherrypy.server, 'using_apache', False):
        return self.skip('skipped due to known Apache differences... ')
    self.getPage('/tinyupload', method='POST', headers=[('Content-Type', 'text/plain'), ('Content-Length', '100')], body='x' * 100)
    self.assertStatus(200)
    self.assertBody('x' * 100)
    self.getPage('/tinyupload', method='POST', headers=[('Content-Type', 'text/plain'), ('Content-Length', '101')], body='x' * 101)
    self.assertStatus(413)
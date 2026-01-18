import os
import cherrypy
from cherrypy.test import helper
def testAdditionalServers(self):
    if self.scheme == 'https':
        return self.skip('not available under ssl')
    self.PORT = 9877
    self.getPage('/')
    self.assertBody(str(self.PORT))
    self.PORT = 9878
    self.getPage('/')
    self.assertBody(str(self.PORT))
import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def testMethodDispatch(self):
    self.getPage('/bymethod')
    self.assertBody("['another']")
    self.assertHeader('Allow', 'GET, HEAD, POST')
    self.getPage('/bymethod', method='HEAD')
    self.assertBody('')
    self.assertHeader('Allow', 'GET, HEAD, POST')
    self.getPage('/bymethod', method='POST', body='thing=one')
    self.assertBody('')
    self.assertHeader('Allow', 'GET, HEAD, POST')
    self.getPage('/bymethod')
    self.assertBody(repr(['another', ntou('one')]))
    self.assertHeader('Allow', 'GET, HEAD, POST')
    self.getPage('/bymethod', method='PUT')
    self.assertErrorPage(405)
    self.assertHeader('Allow', 'GET, HEAD, POST')
    self.getPage('/collection/silly', method='POST')
    self.getPage('/collection', method='GET')
    self.assertBody("['a', 'bit', 'silly']")
    self.getPage('/app')
    self.assertBody('milk')
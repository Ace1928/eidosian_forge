import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def testExpose(self):
    self.getPage('/exposing/base')
    self.assertBody('expose works!')
    self.getPage('/exposing/1')
    self.assertBody('expose works!')
    self.getPage('/exposing/2')
    self.assertBody('expose works!')
    self.getPage('/exposingnew/base')
    self.assertBody('expose works!')
    self.getPage('/exposingnew/1')
    self.assertBody('expose works!')
    self.getPage('/exposingnew/2')
    self.assertBody('expose works!')
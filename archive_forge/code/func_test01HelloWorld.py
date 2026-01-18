import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test01HelloWorld(self):
    self.setup_tutorial('tut01_helloworld', 'HelloWorld')
    self.getPage('/')
    self.assertBody('Hello world!')
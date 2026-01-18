import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def testTreeMounting(self):

    class Root(object):

        @cherrypy.expose
        def hello(self):
            return 'Hello world!'
    a = Application(Root(), '/somewhere')
    self.assertRaises(ValueError, cherrypy.tree.mount, a, '/somewhereelse')
    a = Application(Root(), '/somewhere')
    cherrypy.tree.mount(a, '/somewhere')
    self.getPage('/somewhere/hello')
    self.assertStatus(200)
    del cherrypy.tree.apps['/somewhere']
    cherrypy.tree.mount(a)
    self.getPage('/somewhere/hello')
    self.assertStatus(200)
    a = Application(Root(), script_name=None)
    self.assertRaises(TypeError, cherrypy.tree.mount, a, None)
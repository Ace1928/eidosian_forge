import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test06DefaultMethod(self):
    self.setup_tutorial('tut06_default_method', 'UsersPage')
    self.getPage('/hendrik')
    self.assertBody('Hendrik Mans, CherryPy co-developer & crazy German (<a href="./">back</a>)')
import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test07Sessions(self):
    self.setup_tutorial('tut07_sessions', 'HitCounter')
    self.getPage('/')
    self.assertBody("\n            During your current session, you've viewed this\n            page 1 times! Your life is a patio of fun!\n        ")
    self.getPage('/', self.cookies)
    self.assertBody("\n            During your current session, you've viewed this\n            page 2 times! Your life is a patio of fun!\n        ")
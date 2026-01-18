import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test05DerivedObjects(self):
    self.setup_tutorial('tut05_derived_objects', 'HomePage')
    msg = '\n            <html>\n            <head>\n                <title>Another Page</title>\n            <head>\n            <body>\n            <h2>Another Page</h2>\n\n            <p>\n            And this is the amazing second page!\n            </p>\n\n            </body>\n            </html>\n        '
    msg = msg.replace('</h2>\n\n', '</h2>\n        \n')
    msg = msg.replace('</p>\n\n', '</p>\n        \n')
    self.getPage('/another/')
    self.assertBody(msg)
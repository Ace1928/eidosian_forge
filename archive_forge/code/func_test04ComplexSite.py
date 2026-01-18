import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test04ComplexSite(self):
    self.setup_tutorial('tut04_complex_site', 'root')
    msg = '\n            <p>Here are some extra useful links:</p>\n\n            <ul>\n                <li><a href="http://del.icio.us">del.icio.us</a></li>\n                <li><a href="http://www.cherrypy.dev">CherryPy</a></li>\n            </ul>\n\n            <p>[<a href="../">Return to links page</a>]</p>'
    self.getPage('/links/extra/')
    self.assertBody(msg)
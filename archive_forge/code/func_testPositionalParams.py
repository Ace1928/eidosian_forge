import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
def testPositionalParams(self):
    self.getPage('/dir1/dir2/posparam/18/24/hut/hike')
    self.assertBody('18/24/hut/hike')
    self.getPage('/dir1/dir2/5/3/sir')
    self.assertBody("default for dir1, param is:('dir2', '5', '3', 'sir')")
    self.getPage('/dir1/dir2/script_name/extra/stuff')
    self.assertStatus(404)
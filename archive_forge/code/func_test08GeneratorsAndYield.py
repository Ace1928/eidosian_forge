import sys
import importlib
import cherrypy
from cherrypy.test import helper
def test08GeneratorsAndYield(self):
    self.setup_tutorial('tut08_generators_and_yield', 'GeneratorDemo')
    self.getPage('/')
    self.assertBody('<html><body><h2>Generators rule!</h2><h3>List of users:</h3>Remi<br/>Carlos<br/>Hendrik<br/>Lorenzo Lamas<br/></body></html>')
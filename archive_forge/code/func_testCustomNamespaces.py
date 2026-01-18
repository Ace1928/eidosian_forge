import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def testCustomNamespaces(self):
    self.getPage('/raw/incr?num=12')
    self.assertBody('13')
    self.getPage('/dbscheme')
    self.assertBody('sqlite///memory')
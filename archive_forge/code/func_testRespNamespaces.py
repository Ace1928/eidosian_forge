import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def testRespNamespaces(self):
    self.getPage('/foo/silly')
    self.assertHeader('X-silly', 'sillyval')
    self.assertBody('Hello world')
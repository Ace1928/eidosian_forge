import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
def dbscheme(self):
    return self.db
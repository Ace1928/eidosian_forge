import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'response.headers.X-silly': 'sillyval'})
def silly(self):
    return 'Hello world'
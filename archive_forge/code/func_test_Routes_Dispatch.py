import os
import importlib
import pytest
import cherrypy
from cherrypy.test import helper
def test_Routes_Dispatch(self):
    """Check that routes package based URI dispatching works correctly."""
    self.getPage('/hounslow')
    self.assertStatus('200 OK')
    self.assertBody('Welcome to Hounslow, pop. 10000')
    self.getPage('/foo')
    self.assertStatus('404 Not Found')
    self.getPage('/surbiton')
    self.assertStatus('200 OK')
    self.assertBody('Welcome to Surbiton, pop. 10000')
    self.getPage('/surbiton', method='POST', body='pop=1327')
    self.assertStatus('200 OK')
    self.assertBody('OK')
    self.getPage('/surbiton')
    self.assertStatus('200 OK')
    self.assertHeader('Content-Language', 'en-GB')
    self.assertBody('Welcome to Surbiton, pop. 1327')
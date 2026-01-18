import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_expose_decorator(self):
    self.getPage('/expose_dec/no_call')
    self.assertStatus(200)
    self.assertBody('Mr E. R. Bradshaw')
    self.getPage('/expose_dec/call_empty')
    self.assertStatus(200)
    self.assertBody('Mrs. B.J. Smegma')
    self.getPage('/expose_dec/call_alias')
    self.assertStatus(200)
    self.assertBody('Mr Nesbitt')
    self.getPage('/expose_dec/nesbitt')
    self.assertStatus(200)
    self.assertBody('Mr Nesbitt')
    self.getPage('/expose_dec/alias1')
    self.assertStatus(200)
    self.assertBody('Mr Ken Andrews')
    self.getPage('/expose_dec/alias2')
    self.assertStatus(200)
    self.assertBody('Mr Ken Andrews')
    self.getPage('/expose_dec/andrews')
    self.assertStatus(200)
    self.assertBody('Mr Ken Andrews')
    self.getPage('/expose_dec/alias3')
    self.assertStatus(200)
    self.assertBody('Mr. and Mrs. Watson')
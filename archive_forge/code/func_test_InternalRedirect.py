import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_InternalRedirect(self):
    self.getPage('/internalredirect/')
    self.assertBody('hello')
    self.assertStatus(200)
    self.getPage('/internalredirect/petshop?user_id=Sir-not-appearing-in-this-film')
    self.assertBody('0 images for Sir-not-appearing-in-this-film')
    self.assertStatus(200)
    self.getPage('/internalredirect/petshop?user_id=parrot')
    self.assertBody('0 images for slug')
    self.assertStatus(200)
    self.getPage('/internalredirect/petshop', method='POST', body='user_id=terrier')
    self.assertBody('0 images for fish')
    self.assertStatus(200)
    self.getPage('/internalredirect/early_ir', method='POST', body='arg=aha!')
    self.assertBody('Something went horribly wrong.')
    self.assertStatus(200)
    self.getPage('/internalredirect/secure')
    self.assertBody('Please log in')
    self.assertStatus(200)
    self.getPage('/internalredirect/relative?a=3&b=5')
    self.assertBody('a=3&b=5')
    self.assertStatus(200)
    self.getPage('/internalredirect/choke')
    self.assertStatus(200)
    self.assertBody('Something went horribly wrong.')
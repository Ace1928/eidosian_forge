import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def testUnrepr(self):
    self.getPage('/repr?key=neg')
    self.assertBody('-1234')
    self.getPage('/repr?key=filename')
    self.assertBody(repr(os.path.join(sys.prefix, 'hello.py')))
    self.getPage('/repr?key=thing1')
    self.assertBody(repr(cherrypy.lib.httputil.response_codes[404]))
    if not getattr(cherrypy.server, 'using_apache', False):
        self.getPage('/repr?key=thing2')
        from cherrypy.tutorial import thing2
        self.assertBody(repr(thing2))
    self.getPage('/repr?key=complex')
    self.assertBody('(3+2j)')
    self.getPage('/repr?key=mul')
    self.assertBody('18')
    self.getPage('/repr?key=stradd')
    self.assertBody(repr('112233'))
from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def testParams(self):
    self.getPage('/params/?thing=a')
    self.assertBody(repr(ntou('a')))
    self.getPage('/params/?thing=a&thing=b&thing=c')
    self.assertBody(repr([ntou('a'), ntou('b'), ntou('c')]))
    cherrypy.config.update({'request.show_mismatched_params': True})
    self.getPage('/params/?notathing=meeting')
    self.assertInBody('Missing parameters: thing')
    self.getPage('/params/?thing=meeting&notathing=meeting')
    self.assertInBody('Unexpected query string parameters: notathing')
    cherrypy.config.update({'request.show_mismatched_params': False})
    self.getPage('/params/?notathing=meeting')
    self.assertInBody('Not Found')
    self.getPage('/params/?thing=meeting&notathing=meeting')
    self.assertInBody('Not Found')
    self.getPage('/params/%d4%20%e3/cheese?Gruy%E8re=Bulgn%e9ville')
    self.assertBody('args: %s kwargs: %s' % (('Ô ã', 'cheese'), [('Gruyère', ntou('Bulgnéville'))]))
    self.getPage('/params/code?url=http%3A//cherrypy.dev/index%3Fa%3D1%26b%3D2')
    self.assertBody('args: %s kwargs: %s' % (('code',), [('url', ntou('http://cherrypy.dev/index?a=1&b=2'))]))
    self.getPage('/params/ismap?223,114')
    self.assertBody('Coordinates: 223, 114')
    self.getPage('/params/dictlike?a[1]=1&a[2]=2&b=foo&b[bar]=baz')
    self.assertBody('args: %s kwargs: %s' % (('dictlike',), [('a[1]', ntou('1')), ('a[2]', ntou('2')), ('b', ntou('foo')), ('b[bar]', ntou('baz'))]))
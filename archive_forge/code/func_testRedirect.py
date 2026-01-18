import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def testRedirect(self):
    self.getPage('/redirect/')
    self.assertBody('child')
    self.assertStatus(200)
    self.getPage('/redirect/by_code?code=300')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(300)
    self.getPage('/redirect/by_code?code=301')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(301)
    self.getPage('/redirect/by_code?code=302')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(302)
    self.getPage('/redirect/by_code?code=303')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(303)
    self.getPage('/redirect/by_code?code=307')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(307)
    self.getPage('/redirect/by_code?code=308')
    self.assertMatchesBody('<a href=([\'\\"])(.*)somewhere%20else\\1>\\2somewhere%20else</a>')
    self.assertStatus(308)
    self.getPage('/redirect/nomodify')
    self.assertBody('')
    self.assertStatus(304)
    self.getPage('/redirect/proxy')
    self.assertBody('')
    self.assertStatus(305)
    self.getPage('/redirect/error/')
    self.assertStatus(('302 Found', '303 See Other'))
    self.assertInBody('/errpage')
    self.getPage('/redirect/stringify', protocol='HTTP/1.0')
    self.assertStatus(200)
    self.assertBody("(['%s/'], 302)" % self.base())
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.getPage('/redirect/stringify', protocol='HTTP/1.1')
        self.assertStatus(200)
        self.assertBody("(['%s/'], 303)" % self.base())
    frag = 'foo'
    self.getPage('/redirect/fragment/%s' % frag)
    self.assertMatchesBody('<a href=([\'\\"])(.*)\\/some\\/url\\#%s\\1>\\2\\/some\\/url\\#%s</a>' % (frag, frag))
    loc = self.assertHeader('Location')
    assert loc.endswith('#%s' % frag)
    self.assertStatus(('302 Found', '303 See Other'))
    self.getPage('/redirect/custom?code=303&url=/foobar/%0d%0aSet-Cookie:%20somecookie=someval')
    self.assertStatus(303)
    loc = self.assertHeader('Location')
    assert 'Set-Cookie' in loc
    self.assertNoHeader('Set-Cookie')

    def assertValidXHTML():
        from xml.etree import ElementTree
        try:
            ElementTree.fromstring('<html><body>%s</body></html>' % self.body)
        except ElementTree.ParseError:
            self._handlewebError('automatically generated redirect did not generate well-formed html')
    self.getPage('/redirect/by_code?code=303')
    self.assertStatus(303)
    assertValidXHTML()
    self.getPage('/redirect/url_with_quote')
    self.assertStatus(303)
    assertValidXHTML()
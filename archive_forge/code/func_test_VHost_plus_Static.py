import os
import cherrypy
from cherrypy.test import helper
def test_VHost_plus_Static(self):
    self.getPage('/static/style.css', [('Host', 'www.mydom2.com')])
    self.assertStatus('200 OK')
    self.assertHeader('Content-Type', 'text/css;charset=utf-8')
    self.getPage('/static2/dirback.jpg', [('Host', 'www.mydom2.com')])
    self.assertStatus('200 OK')
    self.assertHeaderIn('Content-Type', ['image/jpeg', 'image/pjpeg'])
    self.getPage('/static2/', [('Host', 'www.mydom2.com')])
    self.assertStatus('200 OK')
    self.assertBody('Hello, world\r\n')
    self.getPage('/static2', [('Host', 'www.mydom2.com')])
    self.assertStatus(301)
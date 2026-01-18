import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
def test_Accept_Tool(self):
    self.getPage('/accept/feed')
    self.assertStatus(200)
    self.assertInBody('<title>Unknown Blog</title>')
    self.getPage('/accept/feed', headers=[('Accept', 'application/atom+xml')])
    self.assertStatus(200)
    self.assertInBody('<title>Unknown Blog</title>')
    self.getPage('/accept/feed', headers=[('Accept', 'application/*')])
    self.assertStatus(200)
    self.assertInBody('<title>Unknown Blog</title>')
    self.getPage('/accept/feed', headers=[('Accept', '*/*')])
    self.assertStatus(200)
    self.assertInBody('<title>Unknown Blog</title>')
    self.getPage('/accept/feed', headers=[('Accept', 'text/html')])
    self.assertErrorPage(406, 'Your client sent this Accept header: text/html. But this resource only emits these media types: application/atom+xml.')
    self.getPage('/accept/')
    self.assertStatus(200)
    self.assertBody('<a href="feed">Atom feed</a>')
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
def test_basic_HTTPMethods(self):
    helper.webtest.methods_with_bodies = ('POST', 'PUT', 'PROPFIND', 'PATCH')
    for m in defined_http_methods:
        self.getPage('/method/', method=m)
        if m == 'HEAD':
            self.assertBody('')
        elif m == 'TRACE':
            self.assertEqual(self.body[:5], b'TRACE')
        else:
            self.assertBody(m)
    self.getPage('/method/parameterized', method='PATCH', body='data=on+top+of+other+things')
    self.assertBody('on top of other things')
    b = 'one thing on top of another'
    h = [('Content-Type', 'text/plain'), ('Content-Length', str(len(b)))]
    self.getPage('/method/request_body', headers=h, method='PATCH', body=b)
    self.assertStatus(200)
    self.assertBody(b)
    b = b'one thing on top of another'
    self.persistent = True
    try:
        conn = self.HTTP_CONN
        conn.putrequest('PATCH', '/method/request_body', skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.putheader('Content-Length', str(len(b)))
        conn.endheaders()
        conn.send(b)
        response = conn.response_class(conn.sock, method='PATCH')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(b)
    finally:
        self.persistent = False
    h = [('Content-Type', 'text/plain')]
    self.getPage('/method/reachable', headers=h, method='PATCH')
    self.assertStatus(411)
    self.getPage('/method/parameterized', method='PUT', body='data=on+top+of+other+things')
    self.assertBody('on top of other things')
    b = 'one thing on top of another'
    h = [('Content-Type', 'text/plain'), ('Content-Length', str(len(b)))]
    self.getPage('/method/request_body', headers=h, method='PUT', body=b)
    self.assertStatus(200)
    self.assertBody(b)
    b = b'one thing on top of another'
    self.persistent = True
    try:
        conn = self.HTTP_CONN
        conn.putrequest('PUT', '/method/request_body', skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.putheader('Content-Length', str(len(b)))
        conn.endheaders()
        conn.send(b)
        response = conn.response_class(conn.sock, method='PUT')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(b)
    finally:
        self.persistent = False
    h = [('Content-Type', 'text/plain')]
    self.getPage('/method/reachable', headers=h, method='PUT')
    self.assertStatus(411)
    b = '<?xml version="1.0" encoding="utf-8" ?>\n\n<propfind xmlns="DAV:"><prop><getlastmodified/></prop></propfind>'
    h = [('Content-Type', 'text/xml'), ('Content-Length', str(len(b)))]
    self.getPage('/method/request_body', headers=h, method='PROPFIND', body=b)
    self.assertStatus(200)
    self.assertBody(b)
    self.getPage('/method/', method='LINK')
    self.assertStatus(405)
    self.getPage('/method/', method='SEARCH')
    self.assertStatus(501)
    self.getPage('/divorce/get?ID=13')
    self.assertBody('Divorce document 13: empty')
    self.assertStatus(200)
    self.getPage('/divorce/', method='GET')
    self.assertBody('<h1>Choose your document</h1>\n<ul>\n</ul>')
    self.assertStatus(200)
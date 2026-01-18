import os
import platform
import threading
import time
from http.client import HTTPConnection
from distutils.spawn import find_executable
import pytest
from path import Path
from more_itertools import consume
import portend
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.lib import sessions
from cherrypy.lib import reprconf
from cherrypy.lib.httputil import response_codes
from cherrypy.test import helper
from cherrypy import _json as json
def test_0_Session(self):
    self.getPage('/set_session_cls/cherrypy.lib.sessions.MemcachedSession')
    self.getPage('/testStr')
    assert self.body == b'1'
    self.getPage('/testGen', self.cookies)
    assert self.body == b'2'
    self.getPage('/testStr', self.cookies)
    assert self.body == b'3'
    self.getPage('/length', self.cookies)
    self.assertErrorPage(500)
    assert b'NotImplementedError' in self.body
    self.getPage('/delkey?key=counter', self.cookies)
    assert self.status_code == 200
    time.sleep(1.25)
    self.getPage('/')
    assert self.body == b'1'
    self.getPage('/keyin?key=counter', self.cookies)
    assert self.body == b'True'
    self.getPage('/delete', self.cookies)
    assert self.body == b'done'
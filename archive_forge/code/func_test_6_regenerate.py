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
def test_6_regenerate(self):
    self.getPage('/testStr')
    id1 = self.cookies[0][1].split(';', 1)[0].split('=', 1)[1]
    self.getPage('/regen')
    assert self.body == b'logged in'
    id2 = self.cookies[0][1].split(';', 1)[0].split('=', 1)[1]
    assert id1 != id2
    self.getPage('/testStr')
    id1 = self.cookies[0][1].split(';', 1)[0].split('=', 1)[1]
    self.getPage('/testStr', headers=[('Cookie', 'session_id=maliciousid; expires=Sat, 27 Oct 2017 04:18:28 GMT; Path=/;')])
    id2 = self.cookies[0][1].split(';', 1)[0].split('=', 1)[1]
    assert id1 != id2
    assert id2 != 'maliciousid'
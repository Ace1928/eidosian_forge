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
def test_8_Ram_Cleanup(self):

    def lock():
        s1 = sessions.RamSession()
        s1.acquire_lock()
        time.sleep(1)
        s1.release_lock()
    t = threading.Thread(target=lock)
    t.start()
    start = time.time()
    while not sessions.RamSession.locks and time.time() - start < 5:
        time.sleep(0.01)
    assert len(sessions.RamSession.locks) == 1, 'Lock not acquired'
    s2 = sessions.RamSession()
    s2.clean_up()
    msg = 'Clean up should not remove active lock'
    assert len(sessions.RamSession.locks) == 1, msg
    t.join()
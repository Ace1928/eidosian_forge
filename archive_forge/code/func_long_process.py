import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
@cherrypy.expose
def long_process(self, seconds='1'):
    try:
        self.longlock.acquire()
        time.sleep(float(seconds))
    finally:
        self.longlock.release()
    return 'success!'
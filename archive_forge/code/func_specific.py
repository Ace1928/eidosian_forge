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
@cherrypy.config(**{'tools.expires.secs': 86400})
def specific(self):
    cherrypy.response.headers['Etag'] = 'need_this_to_make_me_cacheable'
    return 'I am being specific'
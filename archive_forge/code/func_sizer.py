import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
@cherrypy.expose
def sizer(self, size):
    resp = size_cache.get(size, None)
    if resp is None:
        size_cache[size] = resp = 'X' * int(size)
    return resp
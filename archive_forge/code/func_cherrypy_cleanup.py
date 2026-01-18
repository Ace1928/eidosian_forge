import cherrypy
import io
import logging
import os
import re
import sys
from more_itertools import always_iterable
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
def cherrypy_cleanup(data):
    engine.exit()
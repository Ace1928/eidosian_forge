import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def petshop(self, user_id):
    if user_id == 'parrot':
        raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=slug')
    elif user_id == 'terrier':
        raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=fish')
    else:
        raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=%s' % str(user_id))
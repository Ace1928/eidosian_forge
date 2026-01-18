import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def return_dict(self):
    return dict(a=1, b=2, c=3)
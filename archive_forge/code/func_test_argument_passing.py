import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def test_argument_passing(self, num):
    return num * 2
import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def return_composite(self):
    return (dict(a=1, z=26), 'hi', ['welcome', 'friend'])
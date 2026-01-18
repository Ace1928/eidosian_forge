import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def return_datetime(self):
    return DateTime((2003, 10, 7, 8, 1, 0, 1, 280, -1))
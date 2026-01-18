import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
@cherrypy.expose
def return_single_item_list(self):
    return [42]
import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
@cherrypy.expose
def slashes(self):
    cherrypy.request.request_line = 'GET /slashed\\path HTTP/1.1'
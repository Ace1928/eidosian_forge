import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
@pytest.fixture
def configure_server(access_log_file, error_log_file):

    class Root:

        @cherrypy.expose
        def index(self):
            return 'hello'

        @cherrypy.expose
        def uni_code(self):
            cherrypy.request.login = tartaros
            cherrypy.request.remote.name = erebos

        @cherrypy.expose
        def slashes(self):
            cherrypy.request.request_line = 'GET /slashed\\path HTTP/1.1'

        @cherrypy.expose
        def whitespace(self):
            cherrypy.request.headers['User-Agent'] = 'Browzuh (1.0\r\n\t\t.3)'

        @cherrypy.expose
        def as_string(self):
            return 'content'

        @cherrypy.expose
        def as_yield(self):
            yield 'content'

        @cherrypy.expose
        @cherrypy.config(**{'tools.log_tracebacks.on': True})
        def error(self):
            raise ValueError()
    root = Root()
    cherrypy.config.reset()
    cherrypy.config.update({'server.socket_host': webtest.WebCase.HOST, 'server.socket_port': webtest.WebCase.PORT, 'server.protocol_version': webtest.WebCase.PROTOCOL, 'environment': 'test_suite'})
    cherrypy.config.update({'log.error_file': str(error_log_file), 'log.access_file': str(access_log_file)})
    cherrypy.tree.mount(root)
import os
import sys
import time
import cherrypy
@cherrypy.engine.subscribe('start', priority=6)
def log_test_case_name():
    if cherrypy.config.get('test_case_name', False):
        cherrypy.log('STARTED FROM: %s' % cherrypy.config.get('test_case_name'))
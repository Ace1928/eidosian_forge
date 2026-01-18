import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
def run_standard_benchmarks():
    print('')
    print('Client Thread Report (1000 requests, 14 byte response body, %s server threads):' % cherrypy.server.thread_pool)
    print_report(thread_report())
    print('')
    print('Client Thread Report (1000 requests, 14 bytes via staticdir, %s server threads):' % cherrypy.server.thread_pool)
    print_report(thread_report('%s/static/index.html' % SCRIPT_NAME))
    print('')
    print('Size Report (1000 requests, 50 client threads, %s server threads):' % cherrypy.server.thread_pool)
    print_report(size_report())
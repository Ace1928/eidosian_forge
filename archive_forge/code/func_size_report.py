import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
def size_report(sizes=(10, 100, 1000, 10000, 100000, 100000000), concurrency=50):
    sess = ABSession(concurrency=concurrency)
    attrs, names, patterns = list(zip(*sess.parse_patterns))
    yield (('bytes',) + names)
    for sz in sizes:
        sess.path = '%s/sizer?size=%s' % (SCRIPT_NAME, sz)
        sess.run()
        yield ([sz] + [getattr(sess, attr) for attr in attrs])
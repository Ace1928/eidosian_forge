import os
import re
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.process import servers
from cherrypy.test import helper
def read_process(cmd, args=''):
    pipein, pipeout = os.popen4('%s %s' % (cmd, args))
    try:
        firstline = pipeout.readline()
        if re.search('(not recognized|No such file|not found)', firstline, re.IGNORECASE):
            raise IOError('%s must be on your system path.' % cmd)
        output = firstline + pipeout.read()
    finally:
        pipeout.close()
    return output
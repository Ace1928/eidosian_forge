import sys
import logging
from . import compileUi, loadUi
def on_IOError(self, e):
    """ Handle an IOError exception. """
    sys.stderr.write('Error: %s: "%s"\n' % (e.strerror, e.filename))
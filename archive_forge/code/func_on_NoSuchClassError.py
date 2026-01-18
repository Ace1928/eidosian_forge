import sys
import logging
from . import compileUi, loadUi
def on_NoSuchClassError(self, e):
    """ Handle a NoSuchClassError exception. """
    sys.stderr.write(str(e) + '\n')
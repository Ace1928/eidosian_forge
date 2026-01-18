import sys
import logging
from . import compileUi, loadUi
def on_NoSuchWidgetError(self, e):
    """ Handle a NoSuchWidgetError exception. """
    sys.stderr.write(str(e) + '\n')
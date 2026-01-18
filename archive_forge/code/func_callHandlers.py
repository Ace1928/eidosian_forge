import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def callHandlers(self, record):
    """
        Pass a record to all relevant handlers.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. If no handler was found, output a one-off error
        message to sys.stderr. Stop searching up the hierarchy whenever a
        logger with the "propagate" attribute set to zero is found - that
        will be the last logger whose handlers are called.
        """
    c = self
    found = 0
    while c:
        for hdlr in c.handlers:
            found = found + 1
            if record.levelno >= hdlr.level:
                hdlr.handle(record)
        if not c.propagate:
            c = None
        else:
            c = c.parent
    if found == 0:
        if lastResort:
            if record.levelno >= lastResort.level:
                lastResort.handle(record)
        elif raiseExceptions and (not self.manager.emittedNoHandlerWarning):
            sys.stderr.write('No handlers could be found for logger "%s"\n' % self.name)
            self.manager.emittedNoHandlerWarning = True
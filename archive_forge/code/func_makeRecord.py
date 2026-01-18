import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
    """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
    rv = _logRecordFactory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
    if extra is not None:
        for key in extra:
            if key in ['message', 'asctime'] or key in rv.__dict__:
                raise KeyError('Attempt to overwrite %r in LogRecord' % key)
            rv.__dict__[key] = extra[key]
    return rv
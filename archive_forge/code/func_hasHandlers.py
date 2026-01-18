import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def hasHandlers(self):
    """
        See if the underlying logger has any handlers.
        """
    return self.logger.hasHandlers()
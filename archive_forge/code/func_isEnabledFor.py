import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def isEnabledFor(self, level):
    """
        Is this logger enabled for level 'level'?
        """
    return self.logger.isEnabledFor(level)
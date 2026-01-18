import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def getEffectiveLevel(self):
    """
        Get the effective level for the underlying logger.
        """
    return self.logger.getEffectiveLevel()
import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def removeFilter(self, filter):
    """
        Remove the specified filter from this handler.
        """
    if filter in self.filters:
        self.filters.remove(filter)
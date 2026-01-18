import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def getLoggerClass():
    """
    Return the class to be used when instantiating a logger.
    """
    return _loggerClass
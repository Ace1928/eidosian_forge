import collections
from importlib import util
import inspect
import sys
def exception_name(exc):
    return exc.__class__.__name__
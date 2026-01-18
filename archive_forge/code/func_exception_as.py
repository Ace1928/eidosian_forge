import collections
from importlib import util
import inspect
import sys
def exception_as():
    return sys.exc_info()[1]
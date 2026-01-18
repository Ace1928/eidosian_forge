import functools
import inspect
import logging
import traceback
import wsme.exc
import wsme.types
from wsme import utils
def iswsmefunction(f):
    return hasattr(f, '_wsme_definition')
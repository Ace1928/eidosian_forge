import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def requirements_for(meth):
    """Returns a new dict to be used for all route creation as the
            route options"""
    opts = options.copy()
    if method != 'any':
        opts['conditions'] = {'method': [meth.upper()]}
    return opts
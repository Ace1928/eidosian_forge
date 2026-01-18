import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def make_route(self, *args, **kargs):
    """Make a new Route object

        A subclass can override this method to use a custom Route class.
        """
    return Route(*args, **kargs)
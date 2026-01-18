import collections
import itertools as it
import re
import threading
from repoze.lru import LRUCache
import six
from routes import request_config
from routes.util import (
from routes.route import Route
def strip_slashes(name):
    """Remove slashes from the beginning and end of a part/URL."""
    if name.startswith('/'):
        name = name[1:]
    if name.endswith('/'):
        name = name[:-1]
    return name
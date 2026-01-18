import collections
import os
from urllib import parse
import uuid
import stevedore
from cinderclient import exceptions
def safe_issubclass(*args):
    """Like issubclass, but will just return False if not a class."""
    try:
        if issubclass(*args):
            return True
    except TypeError:
        pass
    return False
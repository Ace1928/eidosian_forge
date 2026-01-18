import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def recursive_subclasses(cls):
    """Yield *cls* and direct and indirect subclasses of *cls*."""
    yield cls
    for subcls in cls.__subclasses__():
        yield from recursive_subclasses(subcls)
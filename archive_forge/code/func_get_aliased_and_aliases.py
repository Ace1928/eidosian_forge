import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def get_aliased_and_aliases(d):
    return {*d, *(alias for aliases in d.values() for alias in aliases)}
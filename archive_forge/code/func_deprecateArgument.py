import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
def deprecateArgument(name, msg, category=UserWarning):
    """Raise a warning about deprecated function argument 'name'."""
    warnings.warn('%r is deprecated; %s' % (name, msg), category=category, stacklevel=3)
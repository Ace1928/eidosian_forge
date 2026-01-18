import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def get_unbound_method_function(func):
    """given unbound method, return underlying function"""
    return func if PY3 else func.__func__
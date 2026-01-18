import json
import warnings
from .base import string_types
def reg(klass):
    """registrator function"""
    for name in aliases:
        register(klass, name)
    return klass
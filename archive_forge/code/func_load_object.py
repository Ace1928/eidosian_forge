import ast
import hashlib
import inspect
import os
import re
import warnings
from collections import deque
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import (
from w3lib.html import replace_entities
from scrapy.item import Item
from scrapy.utils.datatypes import LocalWeakReferencedCache
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.python import flatten, to_unicode
def load_object(path: Union[str, Callable]) -> Any:
    """Load an object given its absolute object path, and return it.

    The object can be the import path of a class, function, variable or an
    instance, e.g. 'scrapy.downloadermiddlewares.redirect.RedirectMiddleware'.

    If ``path`` is not a string, but is a callable object, such as a class or
    a function, then return it as is.
    """
    if not isinstance(path, str):
        if callable(path):
            return path
        raise TypeError(f'Unexpected argument type, expected string or object, got: {type(path)}')
    try:
        dot = path.rindex('.')
    except ValueError:
        raise ValueError(f"Error loading object '{path}': not a full path")
    module, name = (path[:dot], path[dot + 1:])
    mod = import_module(module)
    try:
        obj = getattr(mod, name)
    except AttributeError:
        raise NameError(f"Module '{module}' doesn't define any object named '{name}'")
    return obj
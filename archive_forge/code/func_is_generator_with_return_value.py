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
def is_generator_with_return_value(callable: Callable) -> bool:
    """
    Returns True if a callable is a generator function which includes a
    'return' statement with a value different than None, False otherwise
    """
    if callable in _generator_callbacks_cache:
        return bool(_generator_callbacks_cache[callable])

    def returns_none(return_node: ast.Return) -> bool:
        value = return_node.value
        return value is None or (isinstance(value, ast.NameConstant) and value.value is None)
    if inspect.isgeneratorfunction(callable):
        func = callable
        while isinstance(func, partial):
            func = func.func
        src = inspect.getsource(func)
        pattern = re.compile('(^[\\t ]+)')
        code = pattern.sub('', src)
        match = pattern.match(src)
        if match:
            code = re.sub(f'\n{match.group(0)}', '\n', code)
        tree = ast.parse(code)
        for node in walk_callable(tree):
            if isinstance(node, ast.Return) and (not returns_none(node)):
                _generator_callbacks_cache[callable] = True
                return bool(_generator_callbacks_cache[callable])
    _generator_callbacks_cache[callable] = False
    return bool(_generator_callbacks_cache[callable])
import __future__
import ast
import dis
import inspect
import io
import linecache
import re
import sys
import types
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from itertools import zip_longest
from operator import attrgetter
from pathlib import Path
from threading import RLock
from tokenize import detect_encoding
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Sized, Tuple, \
def attr_names_match(attr, argval):
    """
    Checks that the user-visible attr (from ast) can correspond to
    the argval in the bytecode, i.e. the real attribute fetched internally,
    which may be mangled for private attributes.
    """
    if attr == argval:
        return True
    if not attr.startswith('__'):
        return False
    return bool(re.match('^_\\w+%s$' % attr, argval))
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
@classmethod
def for_filename(cls, filename, module_globals=None, use_cache=True):
    if isinstance(filename, Path):
        filename = str(filename)

    def get_lines():
        return linecache.getlines(cast(str, filename), module_globals)
    entry = linecache.cache.get(filename)
    linecache.checkcache(filename)
    lines = get_lines()
    if entry is not None and (not lines):
        linecache.cache[filename] = entry
        lines = get_lines()
    return cls._for_filename_and_lines(filename, tuple(lines))
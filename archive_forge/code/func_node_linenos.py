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
def node_linenos(node):
    if hasattr(node, 'lineno'):
        linenos = []
        if hasattr(node, 'end_lineno') and isinstance(node, ast.expr):
            assert node.end_lineno is not None
            linenos = range(node.lineno, node.end_lineno + 1)
        else:
            linenos = [node.lineno]
        for lineno in linenos:
            yield lineno
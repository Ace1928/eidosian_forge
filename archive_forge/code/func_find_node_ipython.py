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
def find_node_ipython(frame, lasti, stmts, source):
    node = decorator = None
    for stmt in stmts:
        tree = _extract_ipython_statement(stmt)
        try:
            node_finder = NodeFinder(frame, stmts, tree, lasti, source)
            if (node or decorator) and (node_finder.result or node_finder.decorator):
                return (None, None)
            node = node_finder.result
            decorator = node_finder.decorator
        except Exception:
            pass
    return (decorator, node)
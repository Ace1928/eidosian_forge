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
def sections_match(orig_section, dup_section):
    """
    Returns True if the given lists of instructions have matching linenos and opnames.
    """
    return all(((orig_inst.lineno == dup_inst.lineno or 'POP_BLOCK' == orig_inst.opname == dup_inst.opname) and opnames_match(orig_inst, dup_inst) for orig_inst, dup_inst in zip(orig_section, dup_section)))
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
def find_new_matching(orig_section, instructions):
    """
    Yields sections of `instructions` which match `orig_section`.
    The yielded sections include sentinel instructions, but these
    are ignored when checking for matches.
    """
    for start in range(len(instructions) - len(orig_section)):
        indices, dup_section = zip(*islice(non_sentinel_instructions(instructions, start), len(orig_section)))
        if len(dup_section) < len(orig_section):
            return
        if sections_match(orig_section, dup_section):
            yield instructions[start:indices[-1] + 1]
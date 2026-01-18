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
def check_duplicates(original_i, orig_section, original_instructions):
    """
    Returns True if a section of original_instructions starting somewhere other
    than original_i and matching orig_section is found, i.e. orig_section is duplicated.
    """
    for dup_start in range(len(original_instructions)):
        if dup_start == original_i:
            continue
        dup_section = original_instructions[dup_start:dup_start + len(orig_section)]
        if len(dup_section) < len(orig_section):
            return False
        if sections_match(orig_section, dup_section):
            return True
    return False
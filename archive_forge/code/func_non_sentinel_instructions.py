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
def non_sentinel_instructions(instructions, start):
    """
    Yields (index, instruction) pairs excluding the basic
    instructions introduced by the sentinel transformation
    """
    skip_power = False
    for i, inst in islice(enumerate(instructions), start, None):
        if inst.argval == sentinel:
            assert_(inst.opname == 'LOAD_CONST')
            skip_power = True
            continue
        elif skip_power:
            assert_(inst.opname == 'BINARY_POWER')
            skip_power = False
            continue
        yield (i, inst)
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
def handle_jump(original_instructions, original_start, instructions, start):
    """
    Returns the section of instructions starting at `start` and ending
    with a RETURN_VALUE or RAISE_VARARGS instruction.
    There should be a matching section in original_instructions starting at original_start.
    If that section doesn't appear elsewhere in original_instructions,
    then also delete the returned section of instructions.
    """
    for original_j, original_inst, new_j, new_inst in walk_both_instructions(original_instructions, original_start, instructions, start):
        assert_(opnames_match(original_inst, new_inst))
        if original_inst.opname in ('RETURN_VALUE', 'RAISE_VARARGS'):
            inlined = deepcopy(instructions[start:new_j + 1])
            for inl in inlined:
                inl._copied = True
            orig_section = original_instructions[original_start:original_j + 1]
            if not check_duplicates(original_start, orig_section, original_instructions):
                instructions[start:new_j + 1] = []
            return inlined
    return None
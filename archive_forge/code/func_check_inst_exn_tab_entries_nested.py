import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def check_inst_exn_tab_entries_nested(tab: List[InstructionExnTabEntry], indexof) -> None:
    """
    Checks `tab` is a properly sorted list of nested InstructionExnTabEntry's,
    i.e. no entries partially overlap.
    "Properly sorted" means entries are sorted by increasing starts, then
    decreasing ends.
    """
    entry_stack: List[Tuple[int, int]] = []
    for entry in tab:
        key = (indexof[entry.start], indexof[entry.end])
        while entry_stack and entry_stack[-1][1] < key[0]:
            entry_stack.pop()
        if entry_stack:
            assert entry_stack[-1][0] <= key[0] <= key[1] <= entry_stack[-1][1]
        entry_stack.append(key)
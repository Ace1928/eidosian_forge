import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def check_exception_table(tab: List[ExceptionTableEntry]) -> None:
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
    for i in range(len(tab) - 1):
        assert tab[i].start <= tab[i].end and tab[i].end < tab[i + 1].start and (tab[i + 1].start <= tab[i + 1].end)
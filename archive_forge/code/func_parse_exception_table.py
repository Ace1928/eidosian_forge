import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def parse_exception_table(exntab: bytes) -> List[ExceptionTableEntry]:
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    exntab_iter = iter(exntab)
    tab = []
    try:
        while True:
            start = decode_exception_table_varint(exntab_iter) * 2
            length = decode_exception_table_varint(exntab_iter) * 2
            end = start + length - 2
            target = decode_exception_table_varint(exntab_iter) * 2
            dl = decode_exception_table_varint(exntab_iter)
            depth = dl >> 1
            lasti = bool(dl & 1)
            tab.append(ExceptionTableEntry(start, end, target, depth, lasti))
    except StopIteration:
        check_exception_table(tab)
        return tab
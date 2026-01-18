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
def matching_nodes(self, exprs):
    original_instructions = self.get_original_clean_instructions()
    original_index = only((i for i, inst in enumerate(original_instructions) if inst == self.instruction))
    for expr_index, expr in enumerate(exprs):
        setter = get_setter(expr)
        assert setter is not None
        replacement = ast.BinOp(left=expr, op=ast.Pow(), right=ast.Str(s=sentinel))
        ast.fix_missing_locations(replacement)
        setter(replacement)
        try:
            instructions = self.compile_instructions()
        finally:
            setter(expr)
        if sys.version_info >= (3, 10):
            try:
                handle_jumps(instructions, original_instructions)
            except Exception:
                if TESTING or expr_index < len(exprs) - 1:
                    continue
                raise
        indices = [i for i, instruction in enumerate(instructions) if instruction.argval == sentinel]
        for index_num, sentinel_index in enumerate(indices):
            sentinel_index -= index_num * 2
            assert_(instructions.pop(sentinel_index).opname == 'LOAD_CONST')
            assert_(instructions.pop(sentinel_index).opname == 'BINARY_POWER')
        for index_num, sentinel_index in enumerate(indices):
            sentinel_index -= index_num * 2
            new_index = sentinel_index - 1
            if new_index != original_index:
                continue
            original_inst = original_instructions[original_index]
            new_inst = instructions[new_index]
            if original_inst.opname == new_inst.opname in ('CONTAINS_OP', 'IS_OP') and original_inst.arg != new_inst.arg and (original_instructions[original_index + 1].opname != instructions[new_index + 1].opname == 'UNARY_NOT'):
                instructions.pop(new_index + 1)
            if sys.version_info < (3, 10):
                for inst1, inst2 in zip_longest(original_instructions, instructions):
                    assert_(inst1 and inst2 and opnames_match(inst1, inst2))
            yield expr
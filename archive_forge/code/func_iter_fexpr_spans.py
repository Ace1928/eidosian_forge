import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def iter_fexpr_spans(s: str) -> Iterator[Tuple[int, int]]:
    """
    Yields spans corresponding to expressions in a given f-string.
    Spans are half-open ranges (left inclusive, right exclusive).
    Assumes the input string is a valid f-string, but will not crash if the input
    string is invalid.
    """
    stack: List[int] = []
    i = 0
    while i < len(s):
        if s[i] == '{':
            if not stack and i + 1 < len(s) and (s[i + 1] == '{'):
                i += 2
                continue
            stack.append(i)
            i += 1
            continue
        if s[i] == '}':
            if not stack:
                i += 1
                continue
            j = stack.pop()
            if not stack:
                yield (j, i + 1)
            i += 1
            continue
        if stack:
            delim = None
            if s[i:i + 3] in ("'''", '"""'):
                delim = s[i:i + 3]
            elif s[i] in ("'", '"'):
                delim = s[i]
            if delim:
                i += len(delim)
                while i < len(s) and s[i:i + len(delim)] != delim:
                    i += 1
                i += len(delim)
                continue
        i += 1
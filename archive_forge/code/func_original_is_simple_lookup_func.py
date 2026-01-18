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
def original_is_simple_lookup_func(line: Line, index: int, step: Literal[1, -1]) -> bool:
    if step == -1:
        disallowed = {token.RPAR, token.RSQB}
    else:
        disallowed = {token.LPAR, token.LSQB}
    while 0 <= index < len(line.leaves):
        current = line.leaves[index]
        if current.type in disallowed:
            return False
        if current.type not in {token.NAME, token.DOT} or current.value == 'for':
            return True
        index += step
    return True
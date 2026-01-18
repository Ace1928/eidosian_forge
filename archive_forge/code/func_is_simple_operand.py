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
def is_simple_operand(index: int, kind: Literal[1, -1]) -> bool:
    start = line.leaves[index]
    if start.type in {token.NAME, token.NUMBER}:
        return is_simple_lookup(index, kind)
    if start.type in {token.PLUS, token.MINUS, token.TILDE}:
        if line.leaves[index + 1].type in {token.NAME, token.NUMBER}:
            return is_simple_lookup(index + 1, kind=1)
    return False
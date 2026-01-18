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
def do_splitter_match(self, line: Line) -> TMatchResult:
    LL = line.leaves
    if line.leaves[-1].type in OPENING_BRACKETS:
        return TErr('Cannot wrap parens around a line that ends in an opening bracket.')
    string_idx = self._return_match(LL) or self._else_match(LL) or self._assert_match(LL) or self._assign_match(LL) or self._dict_or_lambda_match(LL) or self._prefer_paren_wrap_match(LL)
    if string_idx is not None:
        string_value = line.leaves[string_idx].value
        if not any((char == ' ' or char in SPLIT_SAFE_CHARS for char in string_value)):
            max_string_width = self.line_length - (line.depth + 1) * 4
            if str_width(string_value) > max_string_width:
                if not self.has_custom_splits(string_value):
                    return TErr("We do not wrap long strings in parentheses when the resultant line would still be over the specified line length and can't be split further by StringSplitter.")
        return Ok([string_idx])
    return TErr('This line does not contain any non-atomic strings.')
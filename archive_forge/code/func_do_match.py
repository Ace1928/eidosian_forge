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
def do_match(self, line: Line) -> TMatchResult:
    match_result = self.do_splitter_match(line)
    if isinstance(match_result, Err):
        return match_result
    string_indices = match_result.ok()
    assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
    string_idx = string_indices[0]
    vresult = self._validate(line, string_idx)
    if isinstance(vresult, Err):
        return vresult
    return match_result
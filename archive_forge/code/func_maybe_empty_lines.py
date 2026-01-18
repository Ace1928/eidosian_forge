import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_empty_lines(self, current_line: Line) -> LinesBlock:
    """Return the number of extra empty lines before and after the `current_line`.

        This is for separating `def`, `async def` and `class` with extra empty
        lines (two on module-level).
        """
    form_feed = current_line.depth == 0 and bool(current_line.leaves) and ('\x0c\n' in current_line.leaves[0].prefix)
    before, after = self._maybe_empty_lines(current_line)
    previous_after = self.previous_block.after if self.previous_block else 0
    before = max(0, before - previous_after)
    if self.previous_block and self.previous_block.previous_block is None and (len(self.previous_block.original_line.leaves) == 1) and self.previous_block.original_line.is_docstring and (not (current_line.is_class or current_line.is_def)):
        before = 1
    block = LinesBlock(mode=self.mode, previous_block=self.previous_block, original_line=current_line, before=before, after=after, form_feed=form_feed)
    if current_line.is_comment:
        if self.previous_line is None or (not self.previous_line.is_decorator and (not self.previous_line.is_comment or before) and (self.semantic_leading_comment is None or before)):
            self.semantic_leading_comment = block
    elif not current_line.is_decorator or before:
        self.semantic_leading_comment = None
    self.previous_line = current_line
    self.previous_block = block
    return block
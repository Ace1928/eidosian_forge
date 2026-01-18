import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
@cached_property
def scope_pieces(self) -> List[range]:
    """
        All the pieces (ranges of lines) contained in this object's .scope,
        unless there is no .scope (because the source isn't valid Python syntax)
        in which case it returns all the pieces in the source file, each containing one line.
        """
    if not self.scope:
        return self.source.pieces
    scope_start, scope_end = self.source.line_range(self.scope)
    return [piece for piece in self.source.pieces if scope_start <= piece.start and piece.stop <= scope_end]
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
def executing_piece(self) -> range:
    """
        The piece (range of lines) containing the line currently being executed
        by the interpreter in this frame.
        """
    return only((piece for piece in self.scope_pieces if self.lineno in piece))
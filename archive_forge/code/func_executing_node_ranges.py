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
def executing_node_ranges(self) -> List[RangeInLine]:
    """
        A list of one or zero RangeInLines for the executing node of this frame.
        The list will have one element if the node can be found and it overlaps this line.
        """
    return self._raw_executing_node_ranges(self.frame_info._executing_node_common_indent)
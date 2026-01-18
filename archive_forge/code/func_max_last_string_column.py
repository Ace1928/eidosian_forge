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
def max_last_string_column() -> int:
    """
            Returns:
                The max allowed width of the string value used for the last
                line we will construct.  Note that this value means the width
                rather than the number of characters (e.g., many East Asian
                characters expand to two columns).
            """
    result = self.line_length
    result -= line.depth * 4
    result -= 1 if ends_with_comma else 0
    result -= string_op_leaves_length
    return result
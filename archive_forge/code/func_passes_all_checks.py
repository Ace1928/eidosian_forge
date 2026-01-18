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
def passes_all_checks(i: Index) -> bool:
    """
            Returns:
                True iff ALL of the conditions listed in the 'Transformations'
                section of this classes' docstring would be met by returning @i.
            """
    is_space = string[i] == ' '
    is_split_safe = is_valid_index(i - 1) and string[i - 1] in SPLIT_SAFE_CHARS
    is_not_escaped = True
    j = i - 1
    while is_valid_index(j) and string[j] == '\\':
        is_not_escaped = not is_not_escaped
        j -= 1
    is_big_enough = len(string[i:]) >= self.MIN_SUBSTR_SIZE and len(string[:i]) >= self.MIN_SUBSTR_SIZE
    return (is_space or is_split_safe) and is_not_escaped and is_big_enough and (not breaks_unsplittable_expression(i))
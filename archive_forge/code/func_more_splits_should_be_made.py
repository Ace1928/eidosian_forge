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
def more_splits_should_be_made() -> bool:
    """
            Returns:
                True iff `rest_value` (the remaining string value from the last
                split), should be split again.
            """
    if use_custom_breakpoints:
        return len(custom_splits) > 1
    else:
        return str_width(rest_value) > max_last_string_column()
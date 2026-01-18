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
def breaks_unsplittable_expression(i: Index) -> bool:
    """
            Returns:
                True iff returning @i would result in the splitting of an
                unsplittable expression (which is NOT allowed).
            """
    return i in _illegal_split_indices
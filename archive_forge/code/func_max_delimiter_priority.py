from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def max_delimiter_priority(self, exclude: Iterable[LeafID]=()) -> Priority:
    """Return the highest priority of a delimiter found on the line.

        Values are consistent with what `is_split_*_delimiter()` return.
        Raises ValueError on no delimiters.
        """
    return max((v for k, v in self.delimiters.items() if k not in exclude))
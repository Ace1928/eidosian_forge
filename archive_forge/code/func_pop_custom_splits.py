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
def pop_custom_splits(self, string: str) -> List[CustomSplit]:
    """Custom Split Map Getter Method

        Returns:
            * A list of the custom splits that are mapped to @string, if any
              exist.
              OR
            * [], otherwise.

        Side Effects:
            Deletes the mapping between @string and its associated custom
            splits (which are returned to the caller).
        """
    key = self._get_key(string)
    custom_splits = self._CUSTOM_SPLIT_MAP[key]
    del self._CUSTOM_SPLIT_MAP[key]
    return list(custom_splits)
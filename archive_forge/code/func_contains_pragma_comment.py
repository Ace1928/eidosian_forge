import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def contains_pragma_comment(comment_list: List[Leaf]) -> bool:
    """
    Returns:
        True iff one of the comments in @comment_list is a pragma used by one
        of the more common static analysis tools for python (e.g. mypy, flake8,
        pylint).
    """
    for comment in comment_list:
        if comment.value.startswith(('# type:', '# noqa', '# pylint:')):
            return True
    return False
from __future__ import annotations
import re
from typing import Callable, Dict, Iterable, Iterator, Pattern
from typing import Match as RegexMatch
from .regex_parser import (
def contains_variable(node: Node) -> bool:
    if isinstance(node, Regex):
        return False
    elif isinstance(node, Variable):
        return True
    elif isinstance(node, (Lookahead, Repeat)):
        return contains_variable(node.childnode)
    elif isinstance(node, (NodeSequence, AnyNode)):
        return any((contains_variable(child) for child in node.children))
    return False
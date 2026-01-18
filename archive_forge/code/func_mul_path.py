from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def mul_path(p: Union[URIRef, Path], mul: _MulPathMod) -> MulPath:
    """
    cardinality path
    """
    return MulPath(p, mul)
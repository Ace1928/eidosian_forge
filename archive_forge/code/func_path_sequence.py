from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def path_sequence(self: Union[URIRef, Path], other: Union[URIRef, Path]):
    """
    sequence path
    """
    if not isinstance(other, (URIRef, Path)):
        raise Exception('Only URIRefs or Paths can be in paths!')
    return SequencePath(self, other)
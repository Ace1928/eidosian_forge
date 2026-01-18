from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def makeStatement(self, quadruple: Tuple[Optional[Union[Formula, Graph]], Node, Node, Node], why: Optional[Any]=None) -> None:
    f, p, s, o = quadruple
    if hasattr(p, 'formula'):
        raise ParserError('Formula used as predicate')
    s = self.normalise(f, s)
    p = self.normalise(f, p)
    o = self.normalise(f, o)
    if f == self.rootFormula:
        self.graph.add((s, p, o))
    elif isinstance(f, Formula):
        f.quotedgraph.add((s, p, o))
    else:
        f.add((s, p, o))
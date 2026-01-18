from __future__ import annotations
import collections
import functools
import operator
import typing
from functools import reduce
from typing import (
from pyparsing import ParseResults
from rdflib.paths import (
from rdflib.plugins.sparql.operators import TrueFilter, and_
from rdflib.plugins.sparql.operators import simplify as simplifyFilters
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Prologue, Query, Update
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def reorderTriples(l_: Iterable[Tuple[Identifier, Identifier, Identifier]]) -> List[Tuple[Identifier, Identifier, Identifier]]:
    """
    Reorder triple patterns so that we execute the
    ones with most bindings first
    """

    def _addvar(term: str, varsknown: Set[typing.Union[Variable, BNode]]):
        if isinstance(term, (Variable, BNode)):
            varsknown.add(term)
    l_ = [(None, x) for x in l_]
    varsknown: Set[typing.Union[BNode, Variable]] = set()
    varscount: Dict[Identifier, int] = collections.defaultdict(int)
    for t in l_:
        for c in t[1]:
            if isinstance(c, (Variable, BNode)):
                varscount[c] += 1
    i = 0
    while i < len(l_):
        l_[i:] = sorted(((_knownTerms(x[1], varsknown, varscount), x[1]) for x in l_[i:]))
        t = l_[i][0][0]
        j = 0
        while i + j < len(l_) and l_[i + j][0][0] == t:
            for c in l_[i + j][1]:
                _addvar(c, varsknown)
            j += 1
        i += 1
    return [x[1] for x in l_]
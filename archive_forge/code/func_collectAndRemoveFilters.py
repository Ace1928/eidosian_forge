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
def collectAndRemoveFilters(parts: List[CompValue]) -> Optional[Expr]:
    """

    FILTER expressions apply to the whole group graph pattern in which
    they appear.

    http://www.w3.org/TR/sparql11-query/#sparqlCollectFilters
    """
    filters = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p.name == 'Filter':
            filters.append(translateExists(p.expr))
            parts.pop(i)
        else:
            i += 1
    if filters:
        return and_(*filters)
    return None
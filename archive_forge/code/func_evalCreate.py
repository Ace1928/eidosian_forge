from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence
from rdflib.graph import Graph
from rdflib.plugins.sparql.evaluate import evalBGP, evalPart
from rdflib.plugins.sparql.evalutils import _fillTemplate, _join
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenDict, QueryContext, Update
from rdflib.term import Identifier, URIRef, Variable
def evalCreate(ctx: QueryContext, u: CompValue) -> None:
    """
    http://www.w3.org/TR/sparql11-update/#create
    """
    g = ctx.dataset.get_context(u.graphiri)
    if len(g) > 0:
        raise Exception('Graph %s already exists.' % g.identifier)
    raise Exception('Create not implemented!')
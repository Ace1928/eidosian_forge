from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence
from rdflib.graph import Graph
from rdflib.plugins.sparql.evaluate import evalBGP, evalPart
from rdflib.plugins.sparql.evalutils import _fillTemplate, _join
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenDict, QueryContext, Update
from rdflib.term import Identifier, URIRef, Variable
def evalAdd(ctx: QueryContext, u: CompValue) -> None:
    """

    add all triples from src to dst

    http://www.w3.org/TR/sparql11-update/#add
    """
    src, dst = u.graph
    srcg = _graphOrDefault(ctx, src)
    dstg = _graphOrDefault(ctx, dst)
    if srcg.identifier == dstg.identifier:
        return
    dstg += srcg
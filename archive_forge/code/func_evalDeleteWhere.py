from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence
from rdflib.graph import Graph
from rdflib.plugins.sparql.evaluate import evalBGP, evalPart
from rdflib.plugins.sparql.evalutils import _fillTemplate, _join
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenDict, QueryContext, Update
from rdflib.term import Identifier, URIRef, Variable
def evalDeleteWhere(ctx: QueryContext, u: CompValue) -> None:
    """
    http://www.w3.org/TR/sparql11-update/#deleteWhere
    """
    res: Iterator[FrozenDict] = evalBGP(ctx, u.triples)
    for g in u.quads:
        cg = ctx.dataset.get_context(g)
        c = ctx.pushGraph(cg)
        res = _join(res, list(evalBGP(c, u.quads[g])))
    for c in res:
        g = ctx.graph
        g -= _fillTemplate(u.triples, c)
        for g in u.quads:
            cg = ctx.dataset.get_context(c.get(g))
            cg -= _fillTemplate(u.quads[g], c)
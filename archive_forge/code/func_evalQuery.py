from __future__ import annotations
import collections
import itertools
import json as j
import re
from typing import (
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pyparsing import ParseException
from rdflib.graph import Graph
from rdflib.plugins.sparql import CUSTOM_EVALS, parser
from rdflib.plugins.sparql.aggregates import Aggregator
from rdflib.plugins.sparql.evalutils import (
from rdflib.plugins.sparql.parserutils import CompValue, value
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def evalQuery(graph: Graph, query: Query, initBindings: Optional[Mapping[str, Identifier]]=None, base: Optional[str]=None) -> Mapping[Any, Any]:
    """

    .. caution::

        This method can access indirectly requested network endpoints, for
        example, query processing will attempt to access network endpoints
        specified in ``SERVICE`` directives.

        When processing untrusted or potentially malicious queries, measures
        should be taken to restrict network and file access.

        For information on available security measures, see the RDFLib
        :doc:`Security Considerations </security_considerations>`
        documentation.
    """
    initBindings = dict(((Variable(k), v) for k, v in (initBindings or {}).items()))
    ctx = QueryContext(graph, initBindings=initBindings)
    ctx.prologue = query.prologue
    main = query.algebra
    if main.datasetClause:
        if ctx.dataset is None:
            raise Exception("Non-conjunctive-graph doesn't know about " + 'graphs! Try a query without FROM (NAMED).')
        ctx = ctx.clone()
        firstDefault = False
        for d in main.datasetClause:
            if d.default:
                if firstDefault:
                    dg = ctx.dataset.get_context(BNode())
                    ctx = ctx.pushGraph(dg)
                    firstDefault = True
                ctx.load(d.default, default=True)
            elif d.named:
                g = d.named
                ctx.load(g, default=False)
    return evalPart(ctx, main)
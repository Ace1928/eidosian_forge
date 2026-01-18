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
def evalGraph(ctx: QueryContext, part: CompValue) -> Generator[FrozenBindings, None, None]:
    if ctx.dataset is None:
        raise Exception("Non-conjunctive-graph doesn't know about " + 'graphs. Try a query without GRAPH.')
    ctx = ctx.clone()
    graph: Union[str, Path, None, Graph] = ctx[part.term]
    prev_graph = ctx.graph
    if graph is None:
        for graph in ctx.dataset.contexts():
            if graph == ctx.dataset.default_context:
                continue
            c = ctx.pushGraph(graph)
            c = c.push()
            graphSolution = [{part.term: graph.identifier}]
            for x in _join(evalPart(c, part.p), graphSolution):
                x.ctx.graph = prev_graph
                yield x
    else:
        if TYPE_CHECKING:
            assert not isinstance(graph, Graph)
        c = ctx.pushGraph(ctx.dataset.get_context(graph))
        for x in evalPart(c, part.p):
            x.ctx.graph = prev_graph
            yield x
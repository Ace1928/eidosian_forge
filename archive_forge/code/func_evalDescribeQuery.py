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
def evalDescribeQuery(ctx: QueryContext, query) -> Dict[str, Union[str, Graph]]:
    graph = Graph()
    for pfx, ns in ctx.graph.namespaces():
        graph.bind(pfx, ns)
    to_describe = set()
    for iri in query.PV:
        if isinstance(iri, URIRef):
            to_describe.add(iri)
    if query.p is not None:
        bindings = evalPart(ctx, query.p)
        to_describe.update(*(set(binding.values()) for binding in bindings))
    for resource in to_describe:
        ctx.graph.cbd(resource, target_graph=graph)
    res: Dict[str, Union[str, Graph]] = {}
    res['type_'] = 'DESCRIBE'
    res['graph'] = graph
    return res
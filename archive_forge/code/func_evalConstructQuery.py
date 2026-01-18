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
def evalConstructQuery(ctx: QueryContext, query: CompValue) -> Mapping[str, Union[str, Graph]]:
    template = query.template
    if not template:
        template = query.p.p.triples
    graph = Graph()
    for c in evalPart(ctx, query.p):
        graph += _fillTemplate(template, c)
    res: Dict[str, Union[str, Graph]] = {}
    res['type_'] = 'CONSTRUCT'
    res['graph'] = graph
    return res
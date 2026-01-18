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
def evalBGP(ctx: QueryContext, bgp: List[_Triple]) -> Generator[FrozenBindings, None, None]:
    """
    A basic graph pattern
    """
    if not bgp:
        yield ctx.solution()
        return
    s, p, o = bgp[0]
    _s = ctx[s]
    _p = ctx[p]
    _o = ctx[o]
    for ss, sp, so in ctx.graph.triples((_s, _p, _o)):
        if None in (_s, _p, _o):
            c = ctx.push()
        else:
            c = ctx
        if _s is None:
            c[s] = ss
        try:
            if _p is None:
                c[p] = sp
        except AlreadyBound:
            continue
        try:
            if _o is None:
                c[o] = so
        except AlreadyBound:
            continue
        for x in evalBGP(c, bgp[1:]):
            yield x
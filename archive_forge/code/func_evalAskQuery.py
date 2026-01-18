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
def evalAskQuery(ctx: QueryContext, query: CompValue) -> Mapping[str, Union[str, bool]]:
    res: Dict[str, Union[bool, str]] = {}
    res['type_'] = 'ASK'
    res['askAnswer'] = False
    for x in evalPart(ctx, query.p):
        res['askAnswer'] = True
        break
    return res
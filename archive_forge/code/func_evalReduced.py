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
def evalReduced(ctx: QueryContext, part: CompValue) -> Generator[FrozenBindings, None, None]:
    """apply REDUCED to result

    REDUCED is not as strict as DISTINCT, but if the incoming rows were sorted
    it should produce the same result with limited extra memory and time per
    incoming row.
    """
    MAX = 1
    mru_set = set()
    mru_queue: Deque[Any] = collections.deque()
    for row in evalPart(ctx, part.p):
        if row in mru_set:
            mru_queue.remove(row)
        else:
            yield row
            mru_set.add(row)
            if len(mru_set) > MAX:
                mru_set.remove(mru_queue.pop())
        mru_queue.appendleft(row)
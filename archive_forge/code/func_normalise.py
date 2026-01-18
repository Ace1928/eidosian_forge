from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def normalise(self, f: Optional[Formula], n: Union[Tuple[int, str], bool, int, Decimal, float, _AnyT]) -> Union[URIRef, Literal, BNode, _AnyT]:
    if isinstance(n, tuple):
        return URIRef(str(n[1]))
    if isinstance(n, bool):
        s = Literal(str(n).lower(), datatype=BOOLEAN_DATATYPE)
        return s
    if isinstance(n, int) or isinstance(n, long_type):
        s = Literal(str(n), datatype=INTEGER_DATATYPE)
        return s
    if isinstance(n, Decimal):
        value = str(n)
        if value == '-0':
            value = '0'
        s = Literal(value, datatype=DECIMAL_DATATYPE)
        return s
    if isinstance(n, float):
        s = Literal(str(n), datatype=DOUBLE_DATATYPE)
        return s
    if isinstance(f, Formula):
        if n in f.existentials:
            if TYPE_CHECKING:
                assert isinstance(n, URIRef)
            return f.existentials[n]
    return n
from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
def setEvalFn(self, evalfn: Callable[[Any, Any], Any]) -> Comp:
    self.evalfn = evalfn
    return self
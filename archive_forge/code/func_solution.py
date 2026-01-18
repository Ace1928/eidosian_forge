from __future__ import annotations
import collections
import datetime
import itertools
import typing as t
from collections.abc import Mapping, MutableMapping
from typing import (
import isodate
import rdflib.plugins.sparql
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import NamespaceManager
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Identifier, Literal, Node, URIRef, Variable
def solution(self, vars: Optional[Iterable[Variable]]=None) -> FrozenBindings:
    """
        Return a static copy of the current variable bindings as dict
        """
    if vars:
        return FrozenBindings(self, ((k, v) for k, v in self.bindings.items() if k in vars))
    else:
        return FrozenBindings(self, self.bindings.items())
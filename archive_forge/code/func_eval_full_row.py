from __future__ import annotations
from decimal import Decimal
from typing import (
from rdflib.namespace import XSD
from rdflib.plugins.sparql.datatypes import type_promotion
from rdflib.plugins.sparql.evalutils import _eval, _val
from rdflib.plugins.sparql.operators import numeric
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import FrozenBindings, NotBoundError, SPARQLTypeError
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def eval_full_row(self, row: FrozenBindings) -> FrozenBindings:
    return row
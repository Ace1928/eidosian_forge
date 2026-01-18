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
def newBlankNode(self, arg: Optional[Union[Formula, Graph, Any]]=None, uri: Optional[str]=None, why: Optional[Callable[[], None]]=None) -> BNode:
    if isinstance(arg, Formula):
        return arg.newBlankNode(uri)
    elif isinstance(arg, Graph) or arg is None:
        self.counter += 1
        bn = BNode('n%sb%s' % (self.uuid, self.counter))
    else:
        bn = BNode(str(arg[0]).split('#').pop().replace('_', 'b'))
    return bn
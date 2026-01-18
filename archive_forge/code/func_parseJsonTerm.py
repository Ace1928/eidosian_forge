from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def parseJsonTerm(d: Dict[str, str]) -> Identifier:
    """rdflib object (Literal, URIRef, BNode) for the given json-format dict.

    input is like:
      { 'type': 'uri', 'value': 'http://famegame.com/2006/01/username' }
      { 'type': 'literal', 'value': 'drewp' }
    """
    t = d['type']
    if t == 'uri':
        return URIRef(d['value'])
    elif t == 'literal':
        return Literal(d['value'], datatype=d.get('datatype'), lang=d.get('xml:lang'))
    elif t == 'typed-literal':
        return Literal(d['value'], datatype=URIRef(d['datatype']))
    elif t == 'bnode':
        return BNode(d['value'])
    else:
        raise NotImplementedError('json term type %r' % t)
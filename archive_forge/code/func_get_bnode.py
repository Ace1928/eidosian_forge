from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from xml.sax import handler, make_parser
from xml.sax.handler import ErrorHandler
from rdflib.exceptions import ParserError
from rdflib.graph import Graph
from rdflib.namespace import Namespace
from rdflib.parser import InputSource, Parser
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Literal, URIRef
def get_bnode(self, label: str) -> BNode:
    if self.preserve_bnode_ids:
        bn = BNode(label)
    elif label in self.bnode:
        bn = self.bnode[label]
    else:
        bn = BNode(label)
        self.bnode[label] = bn
    return bn
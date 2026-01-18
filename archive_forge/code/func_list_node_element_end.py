from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from urllib.parse import urldefrag, urljoin
from xml.sax import handler, make_parser, xmlreader
from xml.sax.handler import ErrorHandler
from xml.sax.saxutils import escape, quoteattr
from rdflib.exceptions import Error, ParserError
from rdflib.graph import Graph
from rdflib.namespace import RDF, is_ncname
from rdflib.parser import InputSource, Parser
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.term import BNode, Identifier, Literal, URIRef
def list_node_element_end(self, name: Tuple[str, str], qname) -> None:
    current = self.current
    if self.parent.list == RDF.nil:
        list = BNode()
        self.parent.list = list
        self.store.add((self.parent.list, RDF.first, current.subject))
        self.parent.object = list
        self.parent.char = None
    else:
        list = BNode()
        self.store.add((self.parent.list, RDF.rest, list))
        self.store.add((list, RDF.first, current.subject))
        self.parent.list = list
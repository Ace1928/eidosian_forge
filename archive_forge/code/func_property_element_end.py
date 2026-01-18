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
def property_element_end(self, name: Tuple[str, str], qname) -> None:
    current = self.current
    if current.data is not None and current.object is None:
        literalLang = current.language
        if current.datatype is not None:
            literalLang = None
        current.object = Literal(current.data, literalLang, current.datatype)
        current.data = None
    if self.next.end == self.list_node_element_end:
        if current.object != RDF.nil:
            self.store.add((current.list, RDF.rest, RDF.nil))
    if current.object is not None:
        self.store.add((self.parent.subject, current.predicate, current.object))
        if current.id is not None:
            self.add_reified(current.id, (self.parent.subject, current.predicate, current.object))
    current.subject = None
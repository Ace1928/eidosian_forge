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
def document_element_start(self, name: Tuple[str, str], qname, attrs: AttributesImpl) -> None:
    if name[0] and URIRef(''.join(name)) == RDFVOC.RDF:
        next = getattr(self, 'next')
        next.start = self.node_element_start
        next.end = self.node_element_end
    else:
        self.node_element_start(name, qname, attrs)
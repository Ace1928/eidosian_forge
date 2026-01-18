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
def property_element_start(self, name: Tuple[str, str], qname, attrs: AttributesImpl) -> None:
    name, atts = self.convert(name, qname, attrs)
    current = self.current
    absolutize = self.absolutize
    next = getattr(self, 'next')
    object: Optional[_ObjectType] = None
    current.data = None
    current.list = None
    if not name.startswith(str(RDFNS)):
        current.predicate = absolutize(name)
    elif name == RDFVOC.li:
        current.predicate = current.next_li()
    elif name in PROPERTY_ELEMENT_EXCEPTIONS:
        self.error('Invalid property element URI: %s' % name)
    else:
        current.predicate = absolutize(name)
    id = atts.get(RDFVOC.ID, None)
    if id is not None:
        if not is_ncname(id):
            self.error('rdf:ID value is not a value NCName: %s' % id)
        current.id = absolutize('#%s' % id)
    else:
        current.id = None
    resource = atts.get(RDFVOC.resource, None)
    nodeID = atts.get(RDFVOC.nodeID, None)
    parse_type = atts.get(RDFVOC.parseType, None)
    if resource is not None and nodeID is not None:
        self.error('Property element cannot have both rdf:nodeID and rdf:resource')
    if resource is not None:
        object = absolutize(resource)
        next.start = self.node_element_start
        next.end = self.node_element_end
    elif nodeID is not None:
        if not is_ncname(nodeID):
            self.error('rdf:nodeID value is not a valid NCName: %s' % nodeID)
        if self.preserve_bnode_ids is False:
            if nodeID in self.bnode:
                object = self.bnode[nodeID]
            else:
                subject = BNode()
                self.bnode[nodeID] = subject
                object = subject
        else:
            object = subject = BNode(nodeID)
        next.start = self.node_element_start
        next.end = self.node_element_end
    elif parse_type is not None:
        for att in atts:
            if att != RDFVOC.parseType and att != RDFVOC.ID:
                self.error("Property attr '%s' now allowed here" % att)
        if parse_type == 'Resource':
            current.subject = object = BNode()
            current.char = self.property_element_char
            next.start = self.property_element_start
            next.end = self.property_element_end
        elif parse_type == 'Collection':
            current.char = None
            object = current.list = RDF.nil
            next.start = self.node_element_start
            next.end = self.list_node_element_end
        else:
            object = Literal('', datatype=RDFVOC.XMLLiteral)
            current.char = self.literal_element_char
            current.declared = {XMLNS: 'xml'}
            next.start = self.literal_element_start
            next.char = self.literal_element_char
            next.end = self.literal_element_end
        current.object = object
        return
    else:
        object = None
        current.char = self.property_element_char
        next.start = self.node_element_start
        next.end = self.node_element_end
    datatype = current.datatype = atts.get(RDFVOC.datatype, None)
    language = current.language
    if datatype is not None:
        datatype = absolutize(datatype)
    else:
        for att in atts:
            if not att.startswith(str(RDFNS)):
                predicate = absolutize(att)
            elif att in PROPERTY_ELEMENT_ATTRIBUTES:
                continue
            elif att in PROPERTY_ATTRIBUTE_EXCEPTIONS:
                self.error('Invalid property attribute URI: %s' % att)
            else:
                predicate = absolutize(att)
            o: _ObjectType
            if att == RDF.type:
                o = URIRef(atts[att])
            else:
                if datatype is not None:
                    language = None
                o = Literal(atts[att], language, datatype)
            if object is None:
                object = BNode()
            self.store.add((object, predicate, o))
    if object is None:
        current.data = ''
        current.object = None
    else:
        current.data = None
        current.object = object
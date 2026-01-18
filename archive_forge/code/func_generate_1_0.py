from rdflib import BNode
from rdflib import Literal, URIRef
from rdflib import RDF as ns_rdf
from rdflib.term import XSDToPython
from . import IncorrectBlankNodeUsage, IncorrectLiteral, err_no_blank_node
from .utils import has_one_of_attributes, return_XML
import re
def generate_1_0(self):
    """Generate the property object, 1.0 version"""
    datatype = ''
    dtset = False
    if self.node.hasAttribute('datatype'):
        dtset = True
        dt = self.node.getAttribute('datatype')
        if dt != '':
            datatype = self.state.getURI('datatype')
    if self.state.lang != None:
        lang = self.state.lang
    else:
        lang = ''
    if self.node.hasAttribute('content'):
        val = self.node.getAttribute('content')
        if dtset == False:
            obj = Literal(val, lang=lang)
        else:
            obj = self._create_Literal(val, datatype=datatype, lang=lang)
    elif dtset:
        if datatype == XMLLiteral:
            litval = self._get_XML_literal(self.node)
            obj = Literal(litval, datatype=XMLLiteral)
        elif datatype == HTMLLiteral:
            obj = Literal(self._get_HTML_literal(self.node), datatype=HTMLLiteral)
        else:
            obj = self._create_Literal(self._get_literal(self.node), datatype=datatype, lang=lang)
    elif True in [n.nodeType == self.node.ELEMENT_NODE for n in self.node.childNodes]:
        obj = self._create_Literal(self._get_XML_literal(self.node), datatype=XMLLiteral)
    else:
        obj = self._create_Literal(self._get_literal(self.node), lang=lang)
    for prop in self.state.getURI('property'):
        if not isinstance(prop, BNode):
            self.graph.add((self.subject, prop, obj))
        else:
            self.state.options.add_warning(err_no_blank_node % 'property', warning_type=IncorrectBlankNodeUsage, node=self.node.nodeName)
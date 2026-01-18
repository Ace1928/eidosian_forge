import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def canonicalName(term, g):
    normalized_name = classOrIdentifier(term)
    if isinstance(normalized_name, BNode):
        return term
    elif normalized_name.startswith(XSD):
        return str(term)
    elif first(g.triples_choices((normalized_name, [OWL.unionOf, OWL.intersectionOf], None))):
        return repr(term)
    else:
        return str(term.qname)
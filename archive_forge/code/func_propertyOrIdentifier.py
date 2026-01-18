import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def propertyOrIdentifier(thing):
    if isinstance(thing, Property):
        return thing.identifier
    else:
        assert isinstance(thing, URIRef)
        return thing
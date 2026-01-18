import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def generateQName(graph, uri):
    prefix, uri, localname = graph.compute_qname(classOrIdentifier(uri))
    return ':'.join([prefix, localname])
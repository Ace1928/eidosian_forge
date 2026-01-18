import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def clearInDegree(self):
    """
        Remove references to this individual as an object in the
        backing store.
        """
    self.graph.remove((None, None, self.identifier))
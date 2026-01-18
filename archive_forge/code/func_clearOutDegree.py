import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def clearOutDegree(self):
    """
        Remove all statements to this individual as a subject in the
        backing store. Note that this only removes the statements
        themselves, not the blank node closure so there is a chance
        that this will cause orphaned blank nodes to remain in the
        graph.
        """
    self.graph.remove((self.identifier, None, None))
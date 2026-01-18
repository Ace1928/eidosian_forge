import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def restrictionKind(self):
    for s, p, o in self.graph.triples_choices((self.identifier, self.restrictionKinds, None)):
        return p.split(str(OWL))[-1]
    return None
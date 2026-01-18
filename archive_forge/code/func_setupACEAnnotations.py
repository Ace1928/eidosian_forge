import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def setupACEAnnotations(self):
    self.graph.bind('ace', ACE_NS, override=False)
    self.PN_sgprop = Property(ACE_NS.PN_sg, baseType=OWL.AnnotationProperty, graph=self.graph)
    self.CN_sgprop = Property(ACE_NS.CN_sg, baseType=OWL.AnnotationProperty, graph=self.graph)
    self.CN_plprop = Property(ACE_NS.CN_pl, baseType=OWL.AnnotationProperty, graph=self.graph)
    self.tv_sgprop = Property(ACE_NS.TV_sg, baseType=OWL.AnnotationProperty, graph=self.graph)
    self.tv_plprop = Property(ACE_NS.TV_pl, baseType=OWL.AnnotationProperty, graph=self.graph)
    self.tv_vbgprop = Property(ACE_NS.TV_vbg, baseType=OWL.AnnotationProperty, graph=self.graph)
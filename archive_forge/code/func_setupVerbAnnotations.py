import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def setupVerbAnnotations(self, verb_annotations):
    """

        OWL properties map to ACE transitive verbs (TV)

        There are 6 morphological categories that determine the surface form
        of an IRI:

            singular form of a transitive verb (e.g. mans)
            plural form of a transitive verb (e.g. man)
            past participle form a transitive verb (e.g. manned)

            http://attempto.ifi.uzh.ch/ace_lexicon#TV_sg
            http://attempto.ifi.uzh.ch/ace_lexicon#TV_pl
            http://attempto.ifi.uzh.ch/ace_lexicon#TV_vbg

        """
    if isinstance(verb_annotations, tuple):
        tv_sgprop, tv_plprop, tv_vbg = verb_annotations
    else:
        tv_sgprop = verb_annotations
        tv_plprop = verb_annotations
        tv_vbg = verb_annotations
    if tv_sgprop:
        self.tv_sgprop.extent = [(self.identifier, self.handleAnnotation(tv_sgprop))]
    if tv_plprop:
        self.tv_plprop.extent = [(self.identifier, self.handleAnnotation(tv_plprop))]
    if tv_vbg:
        self.tv_vbgprop.extent = [(self.identifier, self.handleAnnotation(tv_vbg))]
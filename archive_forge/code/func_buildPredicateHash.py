from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def buildPredicateHash(self, subject):
    """
        Build a hash key by predicate to a list of objects for the given
        subject
        """
    properties = {}
    for s, p, o in self.store.triples((subject, None, None)):
        oList = properties.get(p, [])
        oList.append(o)
        properties[p] = oList
    return properties
from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def checkSubject(self, subject):
    """Check to see if the subject should be serialized yet"""
    if self.isDone(subject) or subject not in self._subjects or (subject in self._topLevels and self.depth > 1) or (isinstance(subject, URIRef) and self.depth >= self.maxDepth):
        return False
    return True
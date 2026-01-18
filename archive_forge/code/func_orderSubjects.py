from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def orderSubjects(self):
    seen = {}
    subjects = []
    for classURI in self.topClasses:
        members = list(self.store.subjects(RDF.type, classURI))
        members.sort()
        subjects.extend(members)
        for member in members:
            self._topLevels[member] = True
            seen[member] = True
    recursable = [(isinstance(subject, BNode), self._references[subject], subject) for subject in self._subjects if subject not in seen]
    recursable.sort()
    subjects.extend([subject for isbnode, refs, subject in recursable])
    return subjects
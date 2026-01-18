from random import randint
from rdflib.namespace import RDF
from rdflib.term import BNode, URIRef
def append_multiple(self, other):
    """Adding multiple elements to the container to the end which are in python list other"""
    end = self.end()
    container = self.uri
    for item in other:
        end += 1
        self._len += 1
        elem_uri = str(RDF) + '_' + str(end)
        self.graph.add((container, URIRef(elem_uri), item))
    return self
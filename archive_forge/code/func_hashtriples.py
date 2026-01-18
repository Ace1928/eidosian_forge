from itertools import combinations
from rdflib import BNode, Graph
def hashtriples(self):
    for triple in self:
        g = (isinstance(t, BNode) and self.vhash(t) or t for t in triple)
        yield hash(tuple(g))
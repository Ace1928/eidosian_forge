from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def preprocessTriple(self, triple):
    super(LongTurtleSerializer, self).preprocessTriple(triple)
    for i, node in enumerate(triple):
        if node in self.keywords:
            continue
        self.getQName(node, gen_prefix=i == VERB)
        if isinstance(node, Literal) and node.datatype:
            self.getQName(node.datatype, gen_prefix=_GEN_QNAME_FOR_DT)
    p = triple[1]
    if isinstance(p, BNode):
        self._references[p] += 1
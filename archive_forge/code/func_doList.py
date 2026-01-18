from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def doList(self, l_):
    i = 0
    while l_:
        item = self.store.value(l_, RDF.first)
        if item is not None:
            if i == 0:
                self.write(self.indent(1))
            else:
                self.write('\n' + self.indent(1))
            self.path(item, OBJECT, newline=True)
            self.subjectDone(l_)
        l_ = self.store.value(l_, RDF.rest)
        i += 1
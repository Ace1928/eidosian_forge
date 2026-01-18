from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def s_default(self, subject):
    self.write('\n' + self.indent())
    self.path(subject, SUBJECT)
    self.write('\n' + self.indent())
    self.predicateList(subject)
    self.write('\n.')
    return True
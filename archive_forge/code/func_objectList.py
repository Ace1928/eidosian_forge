from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def objectList(self, objects):
    count = len(objects)
    if count == 0:
        return
    depthmod = count == 1 and 0 or 1
    self.depth += depthmod
    first_nl = False
    if count > 1:
        if not isinstance(objects[0], BNode):
            self.write('\n' + self.indent(1))
        first_nl = True
    self.path(objects[0], OBJECT, newline=first_nl)
    for obj in objects[1:]:
        self.write(' ,')
        if not isinstance(obj, BNode):
            self.write('\n' + self.indent(1))
        self.path(obj, OBJECT, newline=True)
    self.depth -= depthmod
from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def p_squared(self, node, position):
    if not isinstance(node, BNode) or node in self._serialized or self._references[node] > 1 or (position == SUBJECT):
        return False
    if self.isValidList(node):
        self.depth += 2
        self.write(' (\n')
        self.depth -= 2
        self.doList(node)
        self.write('\n' + self.indent() + ')')
    else:
        self.subjectDone(node)
        self.write('\n' + self.indent(1) + '[\n')
        self.depth += 1
        self.predicateList(node)
        self.depth -= 1
        self.write('\n' + self.indent(1) + ']')
    return True
from rdflib.graph import Graph
from rdflib.namespace import OWL, Namespace
from rdflib.plugins.serializers.turtle import OBJECT, SUBJECT, TurtleSerializer
def s_clause(self, subject):
    if isinstance(subject, Graph):
        self.write('\n' + self.indent())
        self.p_clause(subject, SUBJECT)
        self.predicateList(subject)
        self.write(' .')
        return True
    else:
        return False
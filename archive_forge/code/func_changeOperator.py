import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def changeOperator(self, newOperator):
    """
        Converts a unionOf / intersectionOf class expression into one
        that instead uses the given operator

        >>> testGraph = Graph()
        >>> Individual.factoryGraph = testGraph
        >>> EX = Namespace("http://example.com/")
        >>> testGraph.bind("ex", EX, override=False)
        >>> fire = Class(EX.Fire)
        >>> water = Class(EX.Water)
        >>> testClass = BooleanClass(members=[fire,water])
        >>> testClass
        ( ex:Fire AND ex:Water )
        >>> testClass.changeOperator(OWL.unionOf)
        >>> testClass
        ( ex:Fire OR ex:Water )
        >>> try:
        ...     testClass.changeOperator(OWL.unionOf)
        ... except Exception as e:
        ...     print(e)  # doctest: +SKIP
        The new operator is already being used!

        """
    assert newOperator != self._operator, 'The new operator is already being used!'
    self.graph.remove((self.identifier, self._operator, self._rdfList.uri))
    self.graph.add((self.identifier, newOperator, self._rdfList.uri))
    self._operator = newOperator
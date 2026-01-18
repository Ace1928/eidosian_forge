from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
def cbd(self, resource: _SubjectType, *, target_graph: Optional[Graph]=None) -> Graph:
    """Retrieves the Concise Bounded Description of a Resource from a Graph

        Concise Bounded Description (CBD) is defined in [1] as:

        Given a particular node (the starting node) in a particular RDF graph (the source graph), a subgraph of that
        particular graph, taken to comprise a concise bounded description of the resource denoted by the starting node,
        can be identified as follows:

            1. Include in the subgraph all statements in the source graph where the subject of the statement is the
                starting node;

            2. Recursively, for all statements identified in the subgraph thus far having a blank node object, include
                in the subgraph all statements in the source graph where the subject of the statement is the blank node
                in question and which are not already included in the subgraph.

            3. Recursively, for all statements included in the subgraph thus far, for all reifications of each statement
                in the source graph, include the concise bounded description beginning from the rdf:Statement node of
                each reification.

        This results in a subgraph where the object nodes are either URI references, literals, or blank nodes not
        serving as the subject of any statement in the graph.

        [1] https://www.w3.org/Submission/CBD/

        :param resource: a URIRef object, of the Resource for queried for
        :param target_graph: Optionally, a graph to add the CBD to; otherwise, a new graph is created for the CBD
        :return: a Graph, subgraph of self if no graph was provided otherwise the provided graph

        """
    if target_graph is None:
        subgraph = Graph()
    else:
        subgraph = target_graph

    def add_to_cbd(uri: _SubjectType) -> None:
        for s, p, o in self.triples((uri, None, None)):
            subgraph.add((s, p, o))
            if type(o) == BNode and (not (o, None, None) in subgraph):
                add_to_cbd(o)
        for s, p, o in self.triples((None, RDF.subject, uri)):
            for s2, p2, o2 in self.triples((s, None, None)):
                subgraph.add((s2, p2, o2))
    add_to_cbd(resource)
    return subgraph
from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def graph_diff(g1: Graph, g2: Graph) -> Tuple[Graph, Graph, Graph]:
    """Returns three sets of triples: "in both", "in first" and "in second"."""
    cg1 = to_canonical_graph(g1)
    cg2 = to_canonical_graph(g2)
    in_both = cg1 * cg2
    in_first = cg1 - cg2
    in_second = cg2 - cg1
    return (in_both, in_first, in_second)
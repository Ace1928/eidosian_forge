from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def distinguish(self, W: 'Color', graph: Graph):
    colors: Dict[str, Color] = {}
    for n in self.nodes:
        new_color: Tuple[ColorItem, ...] = list(self.color)
        for node in W.nodes:
            new_color += [(1, p, W.hash_color()) for s, p, o in graph.triples((n, None, node))]
            new_color += [(W.hash_color(), p, 3) for s, p, o in graph.triples((node, None, n))]
        new_color = tuple(new_color)
        new_hash_color = self.hash_color(new_color)
        if new_hash_color not in colors:
            c = Color([], self.hashfunc, new_color, hash_cache=self._hash_cache)
            colors[new_hash_color] = c
        colors[new_hash_color].nodes.append(n)
    return colors.values()
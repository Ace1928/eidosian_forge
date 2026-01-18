from __future__ import annotations
from calendar import timegm
from os.path import splitext
from time import altzone, gmtime, localtime, time, timezone
from typing import (
from urllib.parse import quote, urlsplit, urlunsplit
import rdflib.graph  # avoid circular dependency
import rdflib.namespace
import rdflib.term
from rdflib.compat import sign
def find_roots(graph: 'Graph', prop: 'rdflib.term.URIRef', roots: Optional[Set['rdflib.term.Node']]=None) -> Set['rdflib.term.Node']:
    """
    Find the roots in some sort of transitive hierarchy.

    find_roots(graph, rdflib.RDFS.subClassOf)
    will return a set of all roots of the sub-class hierarchy

    Assumes triple of the form (child, prop, parent), i.e. the direction of
    RDFS.subClassOf or SKOS.broader

    """
    non_roots: Set[rdflib.term.Node] = set()
    if roots is None:
        roots = set()
    for x, y in graph.subject_objects(prop):
        non_roots.add(x)
        if x in roots:
            roots.remove(x)
        if y not in non_roots:
            roots.add(y)
    return roots
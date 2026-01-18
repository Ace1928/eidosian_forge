from contextlib import contextmanager
from rdflib.graph import Graph
from rdflib.namespace import RDF
from rdflib.term import BNode, Identifier, Literal, URIRef
def cast_identifier(ref, **kws):
    ref = ref or BNode()
    if not isinstance(ref, Identifier):
        ref = URIRef(ref, **kws)
    return ref
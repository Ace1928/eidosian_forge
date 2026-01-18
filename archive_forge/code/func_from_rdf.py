import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
def from_rdf(graph, context_data=None, base=None, use_native_types=False, use_rdf_type=False, auto_compact=False, startnode=None, index=False):
    if not context_data and auto_compact:
        context_data = dict(((pfx, str(ns)) for pfx, ns in graph.namespaces() if pfx and str(ns) != 'http://www.w3.org/XML/1998/namespace'))
    if isinstance(context_data, Context):
        context = context_data
        context_data = context.to_dict()
    else:
        context = Context(context_data, base=base)
    converter = Converter(context, use_native_types, use_rdf_type)
    result = converter.convert(graph)
    if converter.context.active:
        if isinstance(result, list):
            result = {context.get_key(GRAPH): result}
        result[CONTEXT] = context_data
    return result
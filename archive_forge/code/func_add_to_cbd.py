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
def add_to_cbd(uri: _SubjectType) -> None:
    for s, p, o in self.triples((uri, None, None)):
        subgraph.add((s, p, o))
        if type(o) == BNode and (not (o, None, None) in subgraph):
            add_to_cbd(o)
    for s, p, o in self.triples((None, RDF.subject, uri)):
        for s2, p2, o2 in self.triples((s, None, None)):
            subgraph.add((s2, p2, o2))
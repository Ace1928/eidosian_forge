import sys
from io import StringIO, IOBase
import os
import xml.dom.minidom
from urllib.parse import urlparse
import rdflib
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import RDF as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from .extras.httpheader import acceptable_content_type, content_type
from .transform.prototype import handle_prototypes
from .state import ExecutionContext
from .parse import parse_one_node
from .options import Options
from .transform import top_about, empty_safe_curie, vocab_for_role
from .utils import URIOpener
from .host import HostLanguage, MediaTypes, preferred_suffixes, content_to_host_language
def copyGraph(tog, fromg):
    for t in fromg:
        tog.add(t)
    for k, ns in fromg.namespaces():
        tog.bind(k, ns)
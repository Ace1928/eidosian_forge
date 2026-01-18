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
def rdf_from_sources(self, names, outputFormat='turtle', rdfOutput=False):
    """
        Extract and RDF graph from a list of RDFa sources and serialize them in one graph. The sources are parsed, the RDF
        extracted, and serialization is done in the specified format.
        @param names: list of sources, each can be a URI, a file name, or a file-like object
        @keyword outputFormat: serialization format. Can be one of "turtle", "n3", "xml", "pretty-xml", "nt". "xml", "pretty-xml", "json" or "json-ld". "turtle" and "n3", "xml" and "pretty-xml", and "json" and "json-ld" are synonyms, respectively. Note that the JSON-LD serialization works with RDFLib 3.* only.
        @keyword rdfOutput: controls what happens in case an exception is raised. If the value is False, the caller is responsible handling it; otherwise a graph is returned with an error message included in the processor graph
        @type rdfOutput: boolean
        @return: a serialized RDF Graph
        @rtype: string
        """
    outputFormat = pyRdfa._validate_output_format(outputFormat)
    graph = Graph()
    for name in names:
        self.graph_from_source(name, graph, rdfOutput)
    return str(graph.serialize(format=outputFormat), encoding='utf-8')
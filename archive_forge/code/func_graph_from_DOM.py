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
def graph_from_DOM(self, dom, graph=None, pgraph=None):
    """
        Extract the RDF Graph from a DOM tree. This is where the real processing happens. All other methods get down to this
        one, eventually (e.g., after opening a URI and parsing it into a DOM).
        @param dom: a DOM Node element, the top level entry node for the whole tree (i.e., the C{dom.documentElement} is used to initiate processing down the node hierarchy)
        @keyword graph: an RDF Graph (if None, than a new one is created)
        @type graph: rdflib Graph instance.
        @keyword pgraph: an RDF Graph to hold (possibly) the processor graph content. If None, and the error/warning triples are to be generated, they will be added to the returned graph. Otherwise they are stored in this graph.
        @type pgraph: rdflib Graph instance
        @return: an RDF Graph
        @rtype: rdflib Graph instance
        """

    def copyGraph(tog, fromg):
        for t in fromg:
            tog.add(t)
        for k, ns in fromg.namespaces():
            tog.bind(k, ns)
    if graph == None:
        graph = Graph()
    default_graph = Graph()
    topElement = dom.documentElement
    state = ExecutionContext(topElement, default_graph, base=self.required_base if self.required_base != None else '', options=self.options, rdfa_version=self.rdfa_version)
    for trans in self.options.transformers + builtInTransformers:
        trans(topElement, self.options, state)
    self.rdfa_version = state.rdfa_version
    parse_one_node(topElement, default_graph, None, state, [])
    handle_prototypes(default_graph)
    if self.options.vocab_expansion:
        from .rdfs.process import process_rdfa_sem
        process_rdfa_sem(default_graph, self.options)
    if self.options.experimental_features:
        pass
    if self.options.output_default_graph:
        copyGraph(graph, default_graph)
        if self.options.output_processor_graph:
            if pgraph != None:
                copyGraph(pgraph, self.options.processor_graph.graph)
            else:
                copyGraph(graph, self.options.processor_graph.graph)
    elif self.options.output_processor_graph:
        if pgraph != None:
            copyGraph(pgraph, self.options.processor_graph.graph)
        else:
            copyGraph(graph, self.options.processor_graph.graph)
    self.options.reset_processor_graph()
    return graph
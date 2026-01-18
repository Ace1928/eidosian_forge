from .state import ExecutionContext
from .property import ProcessProperty
from .embeddedRDF import handle_embeddedRDF
from .host import HostLanguage, host_dom_transforms
from rdflib import URIRef
from rdflib import BNode
from rdflib import RDF as ns_rdf
from . import IncorrectBlankNodeUsage, err_no_blank_node
from .utils import has_one_of_attributes
def handle_role_attribute(node, graph, state):
    """
    Handling the role attribute, according to http://www.w3.org/TR/role-attribute/#using-role-in-conjunction-with-rdfa
    @param node: the DOM node to handle
    @param graph: the RDF graph
    @type graph: RDFLib's Graph object instance
    @param state: the inherited state (namespaces, lang, etc.)
    @type state: L{state.ExecutionContext}
    """
    if node.hasAttribute('role'):
        if node.hasAttribute('id'):
            i = node.getAttribute('id').strip()
            subject = URIRef(state.base + '#' + i)
        else:
            subject = BNode()
        predicate = URIRef('http://www.w3.org/1999/xhtml/vocab#role')
        for obj in state.getURI('role'):
            graph.add((subject, predicate, obj))
from .state import ExecutionContext
from .property import ProcessProperty
from .embeddedRDF import handle_embeddedRDF
from .host import HostLanguage, host_dom_transforms
from rdflib import URIRef
from rdflib import BNode
from rdflib import RDF as ns_rdf
from . import IncorrectBlankNodeUsage, err_no_blank_node
from .utils import has_one_of_attributes
def lite_check():
    if state.options.check_lite and state.options.host_language in [HostLanguage.html5, HostLanguage.xhtml5, HostLanguage.xhtml]:
        if node.tagName == 'link' and node.hasAttribute('rel') and (state.term_or_curie.CURIE_to_URI(node.getAttribute('rel')) != None):
            state.options.add_warning('In RDFa Lite, attribute @rel in <link> is only used in non-RDFa way (consider using @property)', node=node)
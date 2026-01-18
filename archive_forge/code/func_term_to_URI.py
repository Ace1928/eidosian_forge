import re
from urllib.parse import urlsplit
from rdflib import URIRef
from rdflib import BNode
from rdflib import Namespace
from .utils import quote_URI
from .host import predefined_1_0_rel, warn_xmlns_usage
from . import IncorrectPrefixDefinition, RDFA_VOCAB, UnresolvableReference, PrefixRedefinitionWarning
from . import err_redefining_URI_as_prefix
from . import err_xmlns_deprecated
from . import err_bnode_local_prefix
from . import err_col_local_prefix
from . import err_missing_URI_prefix
from . import err_invalid_prefix
from . import err_no_default_prefix
from . import err_prefix_and_xmlns
from . import err_non_ncname_prefix
from . import err_absolute_reference
from . import err_query_reference
from . import err_fragment_reference
from . import err_prefix_redefinition
def term_to_URI(self, term):
    """A term to URI mapping, where term is a simple string and the corresponding
        URI is defined via the @vocab (ie, default term uri) mechanism. Returns None if term is not defined
        @param term: string
        @return: an RDFLib URIRef instance (or None)
        """
    if len(term) == 0:
        return None
    if termname.match(term):
        if self.default_term_uri != None:
            return URIRef(self.default_term_uri + term)
        if term in self.terms:
            self.graph.bind(XHTML_PREFIX, XHTML_URI)
            return self.terms[term]
        for defined_term in self.terms:
            if term.lower() == defined_term.lower():
                self.graph.bind(XHTML_PREFIX, XHTML_URI)
                return self.terms[defined_term]
    return None
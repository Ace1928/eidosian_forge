from rdflib import URIRef
from rdflib import BNode
from .host import HostLanguage, accept_xml_base, accept_xml_lang, beautifying_prefixes
from .termorcurie import TermOrCurie
from . import UnresolvablePrefix, UnresolvableTerm
from . import err_URI_scheme
from . import err_illegal_safe_CURIE
from . import err_no_CURIE_in_safe_CURIE
from . import err_undefined_terms
from . import err_non_legal_CURIE_ref
from . import err_undefined_CURIE
from urllib.parse import urlparse, urlunparse, urlsplit, urljoin
def getResource(self, *args):
    """Get single resources from several different attributes. The first one that returns a valid URI wins.
        @param args: variable list of attribute names, or a single attribute being a list itself.
        @return: an RDFLib URIRef instance (or None):
        """
    if len(args) == 0:
        return None
    if isinstance(args[0], tuple) or isinstance(args[0], list):
        rargs = args[0]
    else:
        rargs = args
    for resource in rargs:
        uri = self.getURI(resource)
        if uri != None:
            return uri
    return None
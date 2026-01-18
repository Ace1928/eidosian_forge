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
def remove_frag_id(uri):
    """
            The fragment ID for self.base must be removed
            """
    try:
        t = urlparse(uri)
        return urlunparse((t[0], t[1], t[2], t[3], t[4], ''))
    except:
        return uri
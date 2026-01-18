import datetime
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import Graph
from rdflib import RDF as ns_rdf
from .host import HostLanguage, content_to_host_language, predefined_1_0_rel, require_embedded_rdf
from . import ns_xsd, ns_distill, ns_rdfa
from . import RDFA_Error, RDFA_Warning, RDFA_Info
from .transform.lite import lite_prune
def set_host_language(self, content_type):
    """
        Set the host language for processing, based on the recognized types. If this is not a recognized content type,
        it falls back to RDFa core (i.e., XML)
        @param content_type: content type
        @type content_type: string
        """
    if content_type in content_to_host_language:
        self.host_language = content_to_host_language[content_type]
    else:
        self.host_language = HostLanguage.rdfa_core
    if self.host_language in require_embedded_rdf:
        self.embedded_rdf = True
import sys
from rdflib import Namespace
from rdflib import RDF  as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from ..host import MediaTypes
from ..utils import URIOpener
from . import err_outdated_cache
from . import err_unreachable_vocab
from . import err_unparsable_Turtle_vocab
from . import err_unparsable_ntriples_vocab
from . import err_unparsable_rdfa_vocab
from . import err_unrecognised_vocab_type
from .. import VocabReferenceError
from .cache import CachedVocab, xml_application_media_type
from .. import HTTPError, RDFaError
def return_to_cache(msg):
    if newCache:
        options.add_warning(err_unreachable_vocab % uri, warning_type=VocabReferenceError)
    else:
        options.add_warning(err_outdated_cache % uri, warning_type=VocabReferenceError)
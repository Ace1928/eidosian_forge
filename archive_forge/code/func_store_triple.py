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
def store_triple(self, t):
    """
        In contrast to its name, this does not yet add anything to the graph itself, it just stores the tuple in an
        L{internal set<added_triples>}. (It is important for this to be a set: some of the rules in the various closures may
        generate the same tuples several times.) Before adding the tuple to the set, the method checks whether
        the tuple is in the final graph already (if yes, it is not added to the set).

        The set itself is emptied at the start of every processing cycle; the triples are then effectively added to the
        graph at the end of such a cycle. If the set is
        actually empty at that point, this means that the cycle has not added any new triple, and the full processing can stop.

        @param t: the triple to be added to the graph, unless it is already there
        @type t: a 3-element tuple of (s,p,o)
        """
    if t not in self.graph:
        self.added_triples.add(t)
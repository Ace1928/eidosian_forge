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
def process_rdfa_sem(graph, options):
    """
    Expand the graph through the minimal RDFS and OWL rules defined for RDFa.

    The expansion is done in several steps:
     1. the vocabularies are retrieved from the incoming graph (there are RDFa triples generated for that)
     2. all vocabularies are merged into a separate vocabulary graph
     3. the RDFS/OWL expansion is done on the vocabulary graph, to take care of all the subproperty, subclass, etc, chains
     4. the (expanded) vocabulary graph content is added to the incoming graph
     5. the incoming graph is expanded
     6. the triples appearing in the vocabulary graph are removed from the incoming graph, to avoid unnecessary extra triples from the data

    @param graph: an RDFLib Graph instance, to be expanded
    @param options: options as defined for the RDFa run; used to generate warnings
    @type options: L{pyRdfa.Options}
    """
    vocabs = set()
    from ...pyRdfa import RDFA_VOCAB
    for _s, _p, v in graph.triples((None, RDFA_VOCAB, None)):
        vocabs.add(str(v))
    if len(vocabs) >= 0:
        vocab_graph = Graph()
        for uri in vocabs:
            if options.vocab_cache:
                v_graph = CachedVocab(uri, options).graph
            else:
                v_graph, _exp_date = return_graph(uri, options)
            if v_graph != None:
                for t in v_graph:
                    vocab_graph.add(t)
        MiniOWL(vocab_graph, schema_semantics=True).closure()
        for t in vocab_graph:
            graph.add(t)
        MiniOWL(graph).closure()
        for t in vocab_graph:
            graph.remove(t)
    return graph
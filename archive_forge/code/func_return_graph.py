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
def return_graph(uri, options, newCache=False, verify=True):
    """Parse a file, and return an RDFLib Graph. The URI's content type is checked and either one of
    RDFLib's parsers is invoked (for the Turtle, RDF/XML, and N Triple cases) or a separate RDFa processing is invoked
    on the RDFa content.

    The Accept header of the HTTP request gives a preference to Turtle, followed by RDF/XML and then HTML (RDFa), in case content negotiation is used.

    This function is used to retreive the vocabulary file and turn it into an RDFLib graph.

    @param uri: URI for the graph
    @param options: used as a place where warnings can be sent
    @param newCache: in case this is used with caching, whether a new cache is generated; that modifies the warning text
    @param verify: whether the SSL certificate should be verified
    @return: A tuple consisting of an RDFLib Graph instance and an expiration date); None if the dereferencing or the parsing was unsuccessful
    """

    def return_to_cache(msg):
        if newCache:
            options.add_warning(err_unreachable_vocab % uri, warning_type=VocabReferenceError)
        else:
            options.add_warning(err_outdated_cache % uri, warning_type=VocabReferenceError)
    retval = None
    _expiration_date = None
    content = None
    try:
        form = {'Accept': 'text/html;q=0.8, application/xhtml+xml;q=0.8, text/turtle;q=1.0, application/rdf+xml;q=0.9'}
        content = URIOpener(uri, form, verify)
    except HTTPError:
        _t, value, _traceback = sys.exc_info()
        return_to_cache(value)
        return (None, None)
    except RDFaError:
        _t, value, _traceback = sys.exc_info()
        return_to_cache(value)
        return (None, None)
    except Exception:
        _t, value, _traceback = sys.exc_info()
        return_to_cache(value)
        return (None, None)
    expiration_date = content.expiration_date
    if content.content_type == MediaTypes.turtle:
        try:
            retval = Graph()
            retval.parse(content.data, format='n3')
        except:
            _t, value, _traceback = sys.exc_info()
            options.add_warning(err_unparsable_Turtle_vocab % (uri, value))
    elif content.content_type == MediaTypes.rdfxml:
        try:
            retval = Graph()
            retval.parse(content.data)
        except:
            _type, value, _traceback = sys.exc_info()
            options.add_warning(err_unparsable_Turtle_vocab % (uri, value))
    elif content.content_type == MediaTypes.nt:
        try:
            retval = Graph()
            retval.parse(content.data, format='nt')
        except:
            _type, value, _traceback = sys.exc_info()
            options.add_warning(err_unparsable_ntriples_vocab % (uri, value))
    elif content.content_type in [MediaTypes.xhtml, MediaTypes.html, MediaTypes.xml] or xml_application_media_type.match(content.content_type) != None:
        try:
            from .. import pyRdfa
            from .. import Options
            options = Options()
            retval = pyRdfa(options).graph_from_source(content.data)
        except:
            _type, value, _traceback = sys.exc_info()
            options.add_warning(err_unparsable_rdfa_vocab % (uri, value))
    else:
        options.add_warning(err_unrecognised_vocab_type % (uri, content.content_type))
    return (retval, expiration_date)
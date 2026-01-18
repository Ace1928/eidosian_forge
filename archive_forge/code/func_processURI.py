import sys
from io import StringIO, IOBase
import os
import xml.dom.minidom
from urllib.parse import urlparse
import rdflib
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import RDF as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from .extras.httpheader import acceptable_content_type, content_type
from .transform.prototype import handle_prototypes
from .state import ExecutionContext
from .parse import parse_one_node
from .options import Options
from .transform import top_about, empty_safe_curie, vocab_for_role
from .utils import URIOpener
from .host import HostLanguage, MediaTypes, preferred_suffixes, content_to_host_language
def processURI(uri, outputFormat, form={}):
    """The standard processing of an RDFa uri options in a form; used as an entry point from a CGI call.

    The call accepts extra form options (i.e., HTTP GET options) as follows:

     - C{graph=[output|processor|output,processor|processor,output]} specifying which graphs are returned. Default: C{output}
     - C{space_preserve=[true|false]} means that plain literals are normalized in terms of white spaces. Default: C{false}
     - C{rfa_version} provides the RDFa version that should be used for distilling. The string should be of the form "1.0" or "1.1". Default is the highest version the current package implements, currently "1.1"
     - C{host_language=[xhtml,html,xml]} : the host language. Used when files are uploaded or text is added verbatim, otherwise the HTTP return header should be used. Default C{xml}
     - C{embedded_rdf=[true|false]} : whether embedded turtle or RDF/XML content should be added to the output graph. Default: C{false}
     - C{vocab_expansion=[true|false]} : whether the vocabularies should be expanded through the restricted RDFS entailment. Default: C{false}
     - C{vocab_cache=[true|false]} : whether vocab caching should be performed or whether it should be ignored and vocabulary files should be picked up every time. Default: C{false}
     - C{vocab_cache_report=[true|false]} : whether vocab caching details should be reported. Default: C{false}
     - C{vocab_cache_bypass=[true|false]} : whether vocab caches have to be regenerated every time. Default: C{false}
     - C{rdfa_lite=[true|false]} : whether warnings should be generated for non RDFa Lite attribute usage. Default: C{false}
     - C{certifi_verify=[true|false]} : whether the SSL certificate needs to be verified. Default: C{true}

    @param uri: URI to access. Note that the C{text:} and C{uploaded:} fake URI values are treated separately; the former is for textual intput (in which case a StringIO is used to get the data) and the latter is for uploaded file, where the form gives access to the file directly.
    @param outputFormat: serialization format, as defined by the package. Currently "xml", "turtle", "nt", or "json". Default is "turtle", also used if any other string is given.
    @param form: extra call options (from the CGI call) to set up the local options
    @type form: cgi FieldStorage instance
    @return: serialized graph
    @rtype: string
    """

    def _get_option(param, compare_value, default):
        param_old = param.replace('_', '-')
        if param in list(form.keys()):
            val = form.getfirst(param).lower()
            return val == compare_value
        elif param_old in list(form.keys()):
            val = form.getfirst(param_old).lower()
            return val == compare_value
        else:
            return default
    if uri == 'uploaded:':
        stream = form['uploaded'].file
        base = ''
    elif uri == 'text:':
        stream = StringIO(form.getfirst('text'))
        base = ''
    else:
        stream = uri
        base = uri
    if 'rdfa_version' in list(form.keys()):
        rdfa_version = form.getfirst('rdfa_version')
    else:
        rdfa_version = None
    if 'host_language' in list(form.keys()):
        if form.getfirst('host_language').lower() == 'xhtml':
            media_type = MediaTypes.xhtml
        elif form.getfirst('host_language').lower() == 'html':
            media_type = MediaTypes.html
        elif form.getfirst('host_language').lower() == 'svg':
            media_type = MediaTypes.svg
        elif form.getfirst('host_language').lower() == 'atom':
            media_type = MediaTypes.atom
        else:
            media_type = MediaTypes.xml
    else:
        media_type = ''
    transformers = []
    check_lite = 'rdfa_lite' in list(form.keys()) and form.getfirst('rdfa_lite').lower() == 'true'
    from .transform.metaname import meta_transform
    from .transform.OpenID import OpenID_transform
    from .transform.DublinCore import DC_transform
    if 'extras' in list(form.keys()) and form.getfirst('extras').lower() == 'true':
        for t in [OpenID_transform, DC_transform, meta_transform]:
            transformers.append(t)
    else:
        if 'extra-meta' in list(form.keys()) and form.getfirst('extra-meta').lower() == 'true':
            transformers.append(meta_transform)
        if 'extra-openid' in list(form.keys()) and form.getfirst('extra-openid').lower() == 'true':
            transformers.append(OpenID_transform)
        if 'extra-dc' in list(form.keys()) and form.getfirst('extra-dc').lower() == 'true':
            transformers.append(DC_transform)
    output_default_graph = True
    output_processor_graph = False
    a = None
    if 'graph' in list(form.keys()):
        a = form.getfirst('graph').lower()
    elif 'rdfagraph' in list(form.keys()):
        a = form.getfirst('rdfagraph').lower()
    if a != None:
        if a == 'processor':
            output_default_graph = False
            output_processor_graph = True
        elif a == 'processor,output' or a == 'output,processor':
            output_processor_graph = True
    embedded_rdf = _get_option('embedded_rdf', 'true', False)
    space_preserve = _get_option('space_preserve', 'true', True)
    vocab_cache = _get_option('vocab_cache', 'true', True)
    vocab_cache_report = _get_option('vocab_cache_report', 'true', False)
    refresh_vocab_cache = _get_option('vocab_cache_refresh', 'true', False)
    vocab_expansion = _get_option('vocab_expansion', 'true', False)
    certifi_verify = _get_option('certifi_verify', 'true', True)
    if vocab_cache_report:
        output_processor_graph = True
    options = Options(output_default_graph=output_default_graph, output_processor_graph=output_processor_graph, space_preserve=space_preserve, transformers=transformers, vocab_cache=vocab_cache, vocab_cache_report=vocab_cache_report, refresh_vocab_cache=refresh_vocab_cache, vocab_expansion=vocab_expansion, embedded_rdf=embedded_rdf, check_lite=check_lite, certifi_verify=certifi_verify)
    processor = pyRdfa(options=options, base=base, media_type=media_type, rdfa_version=rdfa_version)
    htmlOutput = False
    import html
    try:
        outputFormat = pyRdfa._validate_output_format(outputFormat)
        if outputFormat == 'n3':
            retval = 'Content-Type: text/rdf+n3; charset=utf-8\n'
        elif outputFormat == 'nt' or outputFormat == 'turtle':
            retval = 'Content-Type: text/turtle; charset=utf-8\n'
        elif outputFormat == 'json-ld' or outputFormat == 'json':
            retval = 'Content-Type: application/ld+json; charset=utf-8\n'
        else:
            retval = 'Content-Type: application/rdf+xml; charset=utf-8\n'
        graph = processor.rdf_from_source(stream, outputFormat, rdfOutput='forceRDFOutput' in list(form.keys()) or not htmlOutput)
        retval += '\n'
        retval += graph
        return retval
    except HTTPError:
        _type, h, _traceback = sys.exc_info()
        retval = 'Content-type: text/html; charset=utf-8\nStatus: %s \n\n' % h.http_code
        retval += '<html>\n'
        retval += '<head>\n'
        retval += '<title>HTTP Error in distilling RDFa content</title>\n'
        retval += '</head><body>\n'
        retval += '<h1>HTTP Error in distilling RDFa content</h1>\n'
        retval += '<p>HTTP Error: %s (%s)</p>\n' % (h.http_code, h.msg)
        retval += "<p>On URI: <code>'%s'</code></p>\n" % html.escape(uri)
        retval += '</body>\n'
        retval += '</html>\n'
        return retval
    except:
        _type, value, _traceback = sys.exc_info()
        import traceback
        retval = 'Content-type: text/html; charset=utf-8\nStatus: %s\n\n' % processor.http_status
        retval += '<html>\n'
        retval += '<head>\n'
        retval += '<title>Exception in RDFa processing</title>\n'
        retval += '</head><body>\n'
        retval += '<h1>Exception in distilling RDFa</h1>\n'
        retval += '<pre>\n'
        strio = StringIO()
        traceback.print_exc(file=strio)
        retval += strio.getvalue()
        retval += '</pre>\n'
        retval += '<pre>%s</pre>\n' % value
        retval += '<h1>Distiller request details</h1>\n'
        retval += '<dl>\n'
        if uri == 'text:' and 'text' in form and (form['text'].value != None) and (len(form['text'].value.strip()) != 0):
            retval += '<dt>Text input:</dt><dd>%s</dd>\n' % html.escape(form['text'].value).replace('\n', '<br/>')
        elif uri == 'uploaded:':
            retval += '<dt>Uploaded file</dt>\n'
        else:
            retval += "<dt>URI received:</dt><dd><code>'%s'</code></dd>\n" % html.escape(uri)
        if 'host_language' in list(form.keys()):
            retval += '<dt>Media Type:</dt><dd>%s</dd>\n' % html.escape(media_type)
        if 'graph' in list(form.keys()):
            retval += '<dt>Requested graphs:</dt><dd>%s</dd>\n' % html.escape(form.getfirst('graph').lower())
        else:
            retval += '<dt>Requested graphs:</dt><dd>default</dd>\n'
        retval += '<dt>Output serialization format:</dt><dd> %s</dd>\n' % outputFormat
        if 'space_preserve' in form:
            retval += '<dt>Space preserve:</dt><dd> %s</dd>\n' % html.escape(form['space_preserve'].value)
        retval += '</dl>\n'
        retval += '</body>\n'
        retval += '</html>\n'
        return retval
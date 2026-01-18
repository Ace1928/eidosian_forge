import logging
import sys
from optparse import OptionParser
import rdflib
from rdflib import plugin
from rdflib.graph import ConjunctiveGraph
from rdflib.parser import Parser
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.util import guess_format
def make_option_parser():
    parser_names = _get_plugin_names(Parser)
    serializer_names = _get_plugin_names(Serializer)
    kw_example = 'FORMAT:(+)KW1,-KW2,KW3=VALUE'
    oparser = OptionParser('%prog [-h] [-i INPUT_FORMAT] [-o OUTPUT_FORMAT] ' + '[--ns=PFX=NS ...] [-] [FILE ...]', description=__doc__.strip() + " Reads file system paths, URLs or from stdin if '-' is given. The result is serialized to stdout.", version='%prog ' + '(using rdflib %s)' % rdflib.__version__)
    oparser.add_option('-i', '--input-format', type=str, help='Format of the input document(s). Available input formats are: %s.' % parser_names + ' If no format is given, it will be ' + 'guessed from the file name extension.' + ' Keywords to parser can be given after format like: %s.' % kw_example, metavar='INPUT_FORMAT')
    oparser.add_option('-o', '--output-format', type=str, default=DEFAULT_OUTPUT_FORMAT, help='Format of the graph serialization. Available output formats are: %s.' % serializer_names + " Default format is: '%default'." + ' Keywords to serializer can be given after format like: %s.' % kw_example, metavar='OUTPUT_FORMAT')
    oparser.add_option('--ns', action='append', type=str, help='Register a namespace binding (QName prefix to a base URI). This can be used more than once.', metavar='PREFIX=NAMESPACE')
    oparser.add_option('--no-guess', dest='guess', action='store_false', default=True, help="Don't guess format based on file suffix.")
    oparser.add_option('--no-out', action='store_true', default=False, help="Don't output the resulting graph " + '(useful for checking validity of input).')
    oparser.add_option('-w', '--warn', action='store_true', default=False, help='Output warnings to stderr (by default only critical errors).')
    return oparser
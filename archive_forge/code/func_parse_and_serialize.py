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
def parse_and_serialize(input_files, input_format, guess, outfile, output_format, ns_bindings, store_conn='', store_type=None):
    if store_type:
        store = plugin.get(store_type, Store)()
        store.open(store_conn)
        graph = ConjunctiveGraph(store)
    else:
        store = None
        graph = ConjunctiveGraph()
    for prefix, uri in ns_bindings.items():
        graph.namespace_manager.bind(prefix, uri, override=False)
    for fpath in input_files:
        use_format, kws = _format_and_kws(input_format)
        if fpath == '-':
            fpath = sys.stdin
        elif not input_format and guess:
            use_format = guess_format(fpath) or DEFAULT_INPUT_FORMAT
        graph.parse(fpath, format=use_format, **kws)
    if outfile:
        output_format, kws = _format_and_kws(output_format)
        kws.setdefault('base', None)
        graph.serialize(destination=outfile, format=output_format, **kws)
    if store:
        store.rollback()
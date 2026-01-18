import codecs
import io
import math
import os
import tempfile
from ast import literal_eval
from contextlib import contextmanager
from textwrap import dedent
import pytest
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer
def test_data_types(self):
    data = [True, False, 10 ** 20, -2e+33, "'", '"&&amp;&&#34;"', [{(b'\xfd',): '\x7f', chr(17476): (1, 2)}, (2, '3')]]
    data.append(chr(83012))
    data.append(literal_eval('{2.3j, 1 - 2.3j, ()}'))
    G = nx.Graph()
    G.name = data
    G.graph['data'] = data
    G.add_node(0, int=-1, data={'data': data})
    G.add_edge(0, 0, float=-2.5, data=data)
    gml = '\n'.join(nx.generate_gml(G, stringizer=literal_stringizer))
    G = nx.parse_gml(gml, destringizer=literal_destringizer)
    assert data == G.name
    assert {'name': data, 'data': data} == G.graph
    assert list(G.nodes(data=True)) == [(0, {'int': -1, 'data': {'data': data}})]
    assert list(G.edges(data=True)) == [(0, 0, {'float': -2.5, 'data': data})]
    G = nx.Graph()
    G.graph['data'] = 'frozenset([1, 2, 3])'
    G = nx.parse_gml(nx.generate_gml(G), destringizer=literal_eval)
    assert G.graph['data'] == 'frozenset([1, 2, 3])'
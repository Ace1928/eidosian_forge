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
def test_labels_are_strings(self):
    answer = 'graph [\n  node [\n    id 0\n    label "1203"\n  ]\n]'
    G = nx.Graph()
    G.add_node(1203)
    data = '\n'.join(nx.generate_gml(G, stringizer=literal_stringizer))
    assert data == answer
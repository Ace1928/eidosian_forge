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
def test_label_kwarg(self):
    G = nx.parse_gml(self.simple_data, label='id')
    assert sorted(G.nodes) == [1, 2, 3]
    labels = [G.nodes[n]['label'] for n in sorted(G.nodes)]
    assert labels == ['Node 1', 'Node 2', 'Node 3']
    G = nx.parse_gml(self.simple_data, label=None)
    assert sorted(G.nodes) == [1, 2, 3]
    labels = [G.nodes[n]['label'] for n in sorted(G.nodes)]
    assert labels == ['Node 1', 'Node 2', 'Node 3']
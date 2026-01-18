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
def test_name(self):
    G = nx.parse_gml('graph [ name "x" node [ id 0 label "x" ] ]')
    assert 'x' == G.graph['name']
    G = nx.parse_gml('graph [ node [ id 0 label "x" ] ]')
    assert '' == G.name
    assert 'name' not in G.graph
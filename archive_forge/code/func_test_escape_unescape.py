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
def test_escape_unescape(self):
    gml = 'graph [\n  name "&amp;&#34;&#xf;&#x4444;&#1234567890;&#x1234567890abcdef;&unknown;"\n]'
    G = nx.parse_gml(gml)
    assert '&"\x0f' + chr(17476) + '&#1234567890;&#x1234567890abcdef;&unknown;' == G.name
    gml = '\n'.join(nx.generate_gml(G))
    alnu = '#1234567890;&#38;#x1234567890abcdef'
    answer = 'graph [\n  name "&#38;&#34;&#15;&#17476;&#38;' + alnu + ';&#38;unknown;"\n]'
    assert answer == gml
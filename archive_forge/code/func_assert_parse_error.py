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
def assert_parse_error(gml):
    pytest.raises(nx.NetworkXError, nx.parse_gml, gml)
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
def assert_generate_error(*args, **kwargs):
    pytest.raises(nx.NetworkXError, lambda: list(nx.generate_gml(*args, **kwargs)))
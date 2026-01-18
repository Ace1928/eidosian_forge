import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_exception_for_unsupported_datatype_graph_attr():
    """Test that a detailed exception is raised when an attribute is of a type
    not supported by GraphML, e.g. a list"""
    pytest.importorskip('lxml.etree')
    G = nx.Graph()
    G.graph['my_list_attribute'] = [0, 1, 2]
    fh = io.BytesIO()
    with pytest.raises(TypeError, match='GraphML does not support'):
        nx.write_graphml(G, fh)
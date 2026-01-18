import networkx as nx
from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, PlanarEmbedding
from networkx.classes.reportviews import NodeView
@staticmethod
def on_start_tests(items):
    for item in items:
        assert hasattr(item, 'add_marker')
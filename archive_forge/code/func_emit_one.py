import abc
from cliff.formatters import base
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite.graphml import GraphMLWriter
from networkx.readwrite import json_graph
import networkx as nx
def emit_one(self, column_names, data, stdout, _=None):
    data = {n: i for n, i in zip(column_names, data)}
    self._reformat(data)
    if nx.__version__ >= '2.0':
        graph = json_graph.node_link_graph(data, attrs={'name': 'graph_index'})
    else:
        graph = json_graph.node_link_graph(data)
    self._write_format(graph, stdout)
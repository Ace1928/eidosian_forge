import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_nodes(self, G, graph_element):
    default = G.graph.get('node_default', {})
    for node, data in G.nodes(data=True):
        node_element = self.myElement('node', id=str(node))
        self.add_attributes('node', node_element, data, default)
        graph_element.append(node_element)
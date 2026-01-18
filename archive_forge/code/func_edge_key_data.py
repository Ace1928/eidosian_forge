import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def edge_key_data(G):
    if G.is_multigraph():
        for u, v, key, data in G.edges(data=True, keys=True):
            edge_data = data.copy()
            edge_data.update(key=key)
            edge_id = edge_data.pop('id', None)
            if edge_id is None:
                edge_id = next(self.edge_id)
                while str(edge_id) in self.all_edge_ids:
                    edge_id = next(self.edge_id)
                self.all_edge_ids.add(str(edge_id))
            yield (u, v, edge_id, edge_data)
    else:
        for u, v, data in G.edges(data=True):
            edge_data = data.copy()
            edge_id = edge_data.pop('id', None)
            if edge_id is None:
                edge_id = next(self.edge_id)
                while str(edge_id) in self.all_edge_ids:
                    edge_id = next(self.edge_id)
                self.all_edge_ids.add(str(edge_id))
            yield (u, v, edge_id, edge_data)